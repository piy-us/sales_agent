"""
Phase 2: Resilient Scraper + RAG Chunker
- Reads all_english_urls.txt produced by Phase 1
- Scrapes each URL with retry + exponential backoff
- Auto-detects docs vs marketing pages
- Splits text into overlapping chunks ready for embedding
- Writes chunks to rag_chunks.jsonl  (append mode → safe to resume)
- Tracks progress in scrape_progress.json so you can Ctrl-C and restart
"""

import json
import time
import hashlib
import re
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
from typing import Optional

import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False
    print("⚠  trafilatura not installed — marketing pages will use fallback extractor")
    print("   Run:  pip install trafilatura")

# ── Config ────────────────────────────────────────────────────────────────────

URL_LIST_FILE   = Path("all_english_urls.txt")
CHUNKS_FILE     = Path("rag_chunks.jsonl")        # final RAG output
PROGRESS_FILE   = Path("scrape_progress.json")    # resume state

# Chunking settings
CHUNK_SIZE      = 400    # words per chunk
CHUNK_OVERLAP   = 80     # overlapping words between consecutive chunks

# HTTP settings
REQUEST_TIMEOUT = 20
MAX_RETRIES     = 4
DELAY_SEC       = 0.5    # base delay between requests
BACKOFF_FACTOR  = 2      # exponential backoff multiplier on retry

# Concurrency: single-threaded but fast enough for thousands of pages.
# For multi-threading bump to concurrent.futures (see comment at bottom).

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scrape.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id:    str
    page_id:     str
    url:         str
    title:       str
    domain:      str
    source_type: str        # "docs" | "marketing"
    chunk_index: int
    total_chunks: int
    text:        str
    headings:    list[str]  # h2/h3 headings on the page (for metadata filtering)
    word_count:  int
    scraped_at:  str

# ── Helpers ───────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "GitLabRAGBot/2.0 (enterprise knowledge base builder)"
})

def url_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]

def chunk_id(page_id: str, idx: int) -> str:
    return f"{page_id}_{idx:04d}"

def clean_text(text: str) -> str:
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def split_into_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += size - overlap   # slide forward, keeping overlap
    return chunks

# ── HTTP fetch with retry ─────────────────────────────────────────────────────

def fetch_html(url: str) -> Optional[str]:
    """Fetch URL with exponential backoff. Returns HTML string or None."""
    delay = DELAY_SEC
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(url, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", delay * 4))
                log.warning(f"  429 rate-limited — sleeping {retry_after}s")
                time.sleep(retry_after)
                continue

            if resp.status_code in (301, 302, 303, 307, 308):
                # requests follows redirects automatically; this shouldn't hit
                pass

            resp.raise_for_status()
            return resp.text

        except requests.exceptions.Timeout:
            log.warning(f"  Timeout (attempt {attempt}/{MAX_RETRIES}) — {url}")
        except requests.exceptions.ConnectionError as e:
            log.warning(f"  Connection error (attempt {attempt}/{MAX_RETRIES}) — {e}")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            if status in (404, 410, 403):
                log.info(f"  HTTP {status} — skipping {url}")
                return None   # permanent failure, don't retry
            log.warning(f"  HTTP {status} (attempt {attempt}/{MAX_RETRIES}) — {url}")
        except Exception as e:
            log.warning(f"  Unexpected error (attempt {attempt}/{MAX_RETRIES}) — {e}")

        if attempt < MAX_RETRIES:
            sleep_time = delay * (BACKOFF_FACTOR ** (attempt - 1))
            log.info(f"  Retrying in {sleep_time:.1f}s …")
            time.sleep(sleep_time)

    log.error(f"  FAILED after {MAX_RETRIES} attempts — {url}")
    return None

# ── Content extractors ────────────────────────────────────────────────────────

def extract_docs(url: str, html: str) -> Optional[tuple[str, str, list[str]]]:
    """Returns (title, text, headings) for docs.gitlab.com pages."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["nav", "footer", "header", "aside", "script",
                     "style", ".sidebar", ".breadcrumb", ".feedback",
                     ".toc", ".pagination"]):
        tag.decompose()

    title_tag = soup.find("h1")
    title = (title_tag.get_text(strip=True) if title_tag
             else (soup.title.string.split("|")[0].strip() if soup.title else url))

    main = (soup.find("main") or soup.find("article") or
            soup.find(class_=re.compile(r"content|doc|markdown", re.I)) or
            soup.body)

    if not main:
        return None

    headings = [h.get_text(strip=True) for h in main.find_all(["h2", "h3"])]
    text = clean_text(main.get_text(separator="\n", strip=True))

    return (title, text, headings) if len(text) >= 100 else None


def extract_marketing(url: str, html: str) -> Optional[tuple[str, str, list[str]]]:
    """Returns (title, text, headings) for about.gitlab.com pages."""
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("h1")
    title = (title_tag.get_text(strip=True) if title_tag
             else (soup.title.string.split("|")[0].strip() if soup.title else url))

    headings = [h.get_text(strip=True) for h in soup.find_all(["h2", "h3"])
                if len(h.get_text(strip=True)) > 3]

    if HAS_TRAFILATURA:
        extracted = trafilatura.extract(
            html, url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=True,
        )
        text = clean_text(extracted) if extracted else ""
    else:
        # Fallback: basic BeautifulSoup extraction
        for tag in soup(["nav", "footer", "header", "script", "style",
                         "aside", ".cookie-banner", ".header", ".footer"]):
            tag.decompose()
        body = soup.find("main") or soup.find("body")
        text = clean_text(body.get_text(separator="\n", strip=True)) if body else ""

    return (title, text, headings) if len(text) >= 100 else None


def page_to_chunks(url: str, title: str, text: str, headings: list[str]) -> list[Chunk]:
    """Split a page's text into overlapping RAG chunks."""
    domain = urlparse(url).netloc
    source_type = "docs" if "docs.gitlab.com" in domain else "marketing"
    page_id = url_id(url)
    scraped_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    raw_chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    total = len(raw_chunks)

    return [
        Chunk(
            chunk_id=chunk_id(page_id, i),
            page_id=page_id,
            url=url,
            title=title,
            domain=domain,
            source_type=source_type,
            chunk_index=i,
            total_chunks=total,
            text=chunk_text,
            headings=headings,
            word_count=len(chunk_text.split()),
            scraped_at=scraped_at,
        )
        for i, chunk_text in enumerate(raw_chunks)
    ]

# ── Progress tracking (resume support) ───────────────────────────────────────

def load_progress() -> set[str]:
    """Load set of already-completed URLs from progress file."""
    if not PROGRESS_FILE.exists():
        return set()
    try:
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        return set(data.get("done", []))
    except Exception:
        return set()

def save_progress(done: set[str]):
    PROGRESS_FILE.write_text(
        json.dumps({"done": sorted(done)}, ensure_ascii=False),
        encoding="utf-8"
    )

# ── Main scrape loop ──────────────────────────────────────────────────────────

def scrape_url(url: str) -> list[Chunk]:
    html = fetch_html(url)
    if not html:
        return []

    domain = urlparse(url).netloc
    result = None

    try:
        if "docs.gitlab.com" in domain:
            result = extract_docs(url, html)
        else:
            result = extract_marketing(url, html)
    except Exception:
        log.error(f"  Extraction error for {url}:\n{traceback.format_exc()}")
        return []

    if not result:
        return []

    title, text, headings = result
    return page_to_chunks(url, title, text, headings)


def main():
    # Load URL list
    if not URL_LIST_FILE.exists():
        print(f"ERROR: {URL_LIST_FILE} not found. Run phase1_sitemap_crawler.py first.")
        return

    urls = [u.strip() for u in URL_LIST_FILE.read_text(encoding="utf-8").splitlines()
            if u.strip() and not u.startswith("#")]

    print(f"Total URLs to scrape: {len(urls)}")

    # Resume support
    done = load_progress()
    remaining = [u for u in urls if u not in done]
    print(f"Already done: {len(done)} | Remaining: {len(remaining)}")

    if not remaining:
        print("All URLs already scraped! Delete scrape_progress.json to re-run.")
        return

    # Open chunks file in append mode (safe to resume)
    total_chunks = 0
    total_pages  = 0
    errors       = 0

    with CHUNKS_FILE.open("a", encoding="utf-8") as fh:
        for i, url in enumerate(remaining, 1):
            log.info(f"[{i}/{len(remaining)}] {url}")

            try:
                chunks = scrape_url(url)
            except KeyboardInterrupt:
                log.info("\nInterrupted — saving progress …")
                save_progress(done)
                print(f"Progress saved. Re-run to continue from where you left off.")
                return
            except Exception:
                log.error(f"  Unhandled error:\n{traceback.format_exc()}")
                errors += 1
                done.add(url)   # mark as attempted so we don't retry forever
                continue

            if chunks:
                for chunk in chunks:
                    fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                total_chunks += len(chunks)
                total_pages  += 1
                log.info(f"  ✓ '{chunks[0].title}' — {len(chunks)} chunks")
            else:
                log.info(f"  ✗ no content extracted")

            done.add(url)

            # Save progress every 50 pages
            if i % 50 == 0:
                save_progress(done)
                log.info(f"  Progress saved ({len(done)} URLs done, {total_chunks} chunks so far)")

            time.sleep(DELAY_SEC)

    # Final save
    save_progress(done)

    print(f"\n{'='*60}")
    print(f"Pages scraped   : {total_pages}")
    print(f"Errors/skipped  : {errors}")
    print(f"Total chunks    : {total_chunks}")
    print(f"Output file     : {CHUNKS_FILE}")
    print(f"{'='*60}")
    print("\nChunk schema:")
    print("  chunk_id, page_id, url, title, domain, source_type,")
    print("  chunk_index, total_chunks, text, headings, word_count, scraped_at")


if __name__ == "__main__":
    main()

# ── Optional: multi-threaded version ─────────────────────────────────────────
# Replace the main loop with ThreadPoolExecutor for 5-10x speed:
#
# from concurrent.futures import ThreadPoolExecutor, as_completed
#
# with ThreadPoolExecutor(max_workers=8) as executor:
#     futures = {executor.submit(scrape_url, url): url for url in remaining}
#     for future in as_completed(futures):
#         url = futures[future]
#         chunks = future.result()
#         ...
#
# Note: bump DELAY_SEC to 0 and add rate limiting with a threading.Semaphore.
