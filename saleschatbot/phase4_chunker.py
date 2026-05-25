"""
Phase 3: Chunker
- Reads scraped_pages.jsonl  (web pages — from phase2_scraper.py)
- Reads ppt_chunks.jsonl     (PPT chunks — output of enterprise_pptx_ingestion.py's assemble_chunks())
- Produces chunks.jsonl      (ready to embed + push to Azure AI Search)
- Resume-safe via chunk_progress.json

Every output record matches the agreed index schema exactly:
  id             – unique per chunk  → "{source_id}_chunk_{n}"
  source_id      – groups all chunks of one page/file  → sha256[:16] of url or filepath
  source_type    – "gitlab_docs" | "gitlab_marketing" | "tcs_internal"
  url            – web pages only, null for PPT
  file_name      – PPT only, null for web pages
  title          – page title or PPT deck name
  text           – chunk text (captions already merged in for PPT visual chunks)
  chunk_index    – 0-based position within parent doc
  slide_number   – PPT only, null for web pages
  has_image      – true when an image_blob_url is present
  image_blob_url – blob storage URL for agent tool fetch, null if none
  scraped_at     – ISO timestamp from the source record
"""

import json
import hashlib
import re
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

WEB_INPUT_FILE = Path("scraped_pages.jsonl")   # from phase2_scraper.py
PPT_INPUT_FILE = Path("ppt_chunks.jsonl")       # from enterprise_pptx_ingestion.py
OUTPUT_FILE    = Path("chunks.jsonl")           # → embed → Azure AI Search
PROGRESS_FILE  = Path("chunk_progress.json")

# Chunking knobs for web pages
CHUNK_SIZE_WORDS    = 300   # target words per chunk
CHUNK_OVERLAP_WORDS = 50    # overlap to preserve context across chunk boundaries
MIN_CHUNK_WORDS     = 50    # discard anything shorter than this

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chunk.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Output schema ─────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    id:             str
    source_id:      str
    source_type:    str            # "gitlab_docs" | "gitlab_marketing" | "tcs_internal"
    url:            Optional[str]
    file_name:      Optional[str]
    title:          str
    text:           str
    chunk_index:    int
    slide_number:   Optional[int]
    has_image:      bool
    image_blob_url: Optional[str]
    scraped_at:     str

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_source_id(raw: str) -> str:
    """Stable 16-char ID from any string (URL or file path)."""
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def make_chunk_id(source_id: str, index: int) -> str:
    return f"{source_id}_chunk_{index}"

def split_into_words(text: str) -> list[str]:
    return text.split()

def words_to_text(words: list[str]) -> str:
    return " ".join(words)

# ── Web page chunker ──────────────────────────────────────────────────────────

def chunk_web_page(record: dict) -> list[Chunk]:
    """
    Sliding window over plain text.
    Tries to break at sentence boundaries within the window to avoid
    cutting mid-sentence, then falls back to hard word-count split.
    """
    source_id   = record["page_id"]           # already sha256[:16] from scraper
    source_type = (
        "gitlab_docs"      if record["source_type"] == "docs"
        else "gitlab_marketing"
    )
    url        = record["url"]
    title      = record["title"]
    scraped_at = record["scraped_at"]
    text       = record["text"].strip()

    words  = split_into_words(text)
    chunks = []
    start  = 0
    idx    = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE_WORDS, len(words))

        # try to snap end to a sentence boundary within the last 30 words
        if end < len(words):
            window_text = words_to_text(words[max(end - 30, start): end])
            # find last sentence-ending punctuation in that window
            matches = list(re.finditer(r"[.!?][\s]", window_text))
            if matches:
                last_match_pos = matches[-1].start()
                # how many words into the window does this land?
                prefix_text = window_text[: last_match_pos + 1]
                snap_words  = len(prefix_text.split())
                end = max(end - 30, start) + snap_words

        chunk_text = words_to_text(words[start:end]).strip()

        if len(chunk_text.split()) >= MIN_CHUNK_WORDS:
            chunks.append(Chunk(
                id             = make_chunk_id(source_id, idx),
                source_id      = source_id,
                source_type    = source_type,
                url            = url,
                file_name      = None,
                title          = title,
                text           = chunk_text,
                chunk_index    = idx,
                slide_number   = None,
                has_image      = False,
                image_blob_url = None,
                scraped_at     = scraped_at,
            ))
            idx += 1

        # advance with overlap so context isn't hard-cut
        start = end - CHUNK_OVERLAP_WORDS if end < len(words) else len(words)

    return chunks

# ── PPT chunk normaliser ──────────────────────────────────────────────────────

def normalise_ppt_chunk(ppt_chunk: dict, chunk_index: int) -> Optional[Chunk]:
    """
    PPT chunks come out of assemble_chunks() already split one-per-slide
    (text chunk) or one-per-visual (image/chart chunk).
    We just normalise them into the unified schema — no re-splitting needed.
    """
    meta       = ppt_chunk.get("metadata", {})
    file_name  = meta.get("source_file", "")
    source_id  = make_source_id(file_name)
    text       = ppt_chunk.get("text", "").strip()

    if not text or len(text.split()) < MIN_CHUNK_WORDS:
        return None

    image_uri = ppt_chunk.get("image_uri") or meta.get("image_uri") or None

    # modified_date from pptx core properties as scraped_at stand-in
    scraped_at = meta.get("modified_date") or ""

    return Chunk(
        id             = make_chunk_id(source_id, chunk_index),
        source_id      = source_id,
        source_type    = "tcs_internal",
        url            = None,
        file_name      = file_name,
        title          = meta.get("slide_title") or file_name,
        text           = text,
        chunk_index    = chunk_index,
        slide_number   = meta.get("slide_number"),
        has_image      = image_uri is not None,
        image_blob_url = image_uri,
        scraped_at     = scraped_at,
    )

# ── Progress tracking ─────────────────────────────────────────────────────────

def load_progress() -> set:
    if not PROGRESS_FILE.exists():
        return set()
    try:
        return set(json.loads(PROGRESS_FILE.read_text(encoding="utf-8")).get("done", []))
    except Exception:
        return set()

def save_progress(done: set):
    PROGRESS_FILE.write_text(
        json.dumps({"done": sorted(done)}, ensure_ascii=False),
        encoding="utf-8",
    )

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    done = load_progress()
    total_chunks = 0

    with OUTPUT_FILE.open("a", encoding="utf-8") as out:

        # ── Web pages ─────────────────────────────────────────────────────────
        if WEB_INPUT_FILE.exists():
            web_records = [
                json.loads(line)
                for line in WEB_INPUT_FILE.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            log.info(f"Web pages to chunk: {len(web_records)}")

            for record in web_records:
                page_id = record["page_id"]
                if page_id in done:
                    continue

                try:
                    chunks = chunk_web_page(record)
                except Exception as e:
                    log.error(f"  Failed chunking {record.get('url')}: {e}")
                    done.add(page_id)
                    continue

                for chunk in chunks:
                    out.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

                total_chunks += len(chunks)
                done.add(page_id)
                log.info(f"  ✓ {record['url']} → {len(chunks)} chunks")

            save_progress(done)
        else:
            log.warning(f"{WEB_INPUT_FILE} not found — skipping web pages")

        # ── PPT chunks ────────────────────────────────────────────────────────
        if PPT_INPUT_FILE.exists():
            ppt_records = [
                json.loads(line)
                for line in PPT_INPUT_FILE.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            log.info(f"PPT raw chunks to normalise: {len(ppt_records)}")

            # group by source_file so chunk_index is per-file, not global
            from collections import defaultdict
            by_file: dict[str, list] = defaultdict(list)
            for r in ppt_records:
                fname = r.get("metadata", {}).get("source_file", "unknown")
                by_file[fname].append(r)

            for fname, file_chunks in by_file.items():
                source_id = make_source_id(fname)
                if source_id in done:
                    continue

                saved = 0
                for idx, raw in enumerate(file_chunks):
                    chunk = normalise_ppt_chunk(raw, idx)
                    if chunk:
                        out.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                        saved += 1

                total_chunks += saved
                done.add(source_id)
                log.info(f"  ✓ {fname} → {saved} chunks")

            save_progress(done)
        else:
            log.warning(f"{PPT_INPUT_FILE} not found — skipping PPT chunks")

    print(f"\n{'='*60}")
    print(f"Total chunks written : {total_chunks}  →  {OUTPUT_FILE}")
    print(f"{'='*60}")
    print("Next step: python phase4_embed_and_index.py")


if __name__ == "__main__":
    main()