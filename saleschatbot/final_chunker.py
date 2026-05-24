"""
enterprise_sales_chunker.py
============================================================

Enterprise Sales RAG Chunker

Designed for:
- sales engineering
- enterprise positioning
- pricing conversations
- customer proof points
- solution architecture discussions

Optimized for markdown produced by:
    enterprise_sales_scraper.py

KEY IMPROVEMENTS
----------------
✓ Real markdown-aware chunking
✓ H1/H2 segmentation only
✓ Preserves markdown tables
✓ Preserves bullet lists
✓ Real tokenizer counts (tiktoken)
✓ Paragraph-aware overlap
✓ Larger enterprise-sales chunk sizes
✓ Content-type-aware chunk sizing
✓ Better overlap logic
✓ Cleaner chunk metadata
✓ Better quality filtering
✓ Prevents tiny fragmented chunks

INSTALL
-------
pip install tiktoken

INPUT
-----
gitlab_kb.jsonl

OUTPUT
------
chunked_dataset.jsonl
"""

import json
import re
import hashlib

from pathlib import Path

import tiktoken

# ============================================================
# FILES
# ============================================================

INPUT_FILE = Path("gitlab_kb.jsonl")

OUTPUT_FILE = Path("chunked_dataset.jsonl")

# ============================================================
# TOKENIZER
# ============================================================

ENCODER = tiktoken.get_encoding("cl100k_base")

# ============================================================
# CHUNK CONFIG
# ============================================================
#
# Sales RAG benefits from:
# - larger chunks
# - more context continuity
# - preserving narratives
#
# ============================================================

CHUNK_CONFIG = {

    # --------------------------------------------------------
    # about.gitlab.com
    # --------------------------------------------------------

    ("about", "pricing"): {
        "size": 700,
        "overlap": 120,
    },

    ("about", "solutions"): {
        "size": 900,
        "overlap": 150,
    },

    ("about", "customers"): {
        "size": 1200,
        "overlap": 200,
    },

    ("about", "topics"): {
        "size": 900,
        "overlap": 150,
    },

    ("about", "*"): {
        "size": 850,
        "overlap": 150,
    },

    # --------------------------------------------------------
    # docs.gitlab.com
    # --------------------------------------------------------

    ("docs", "subscriptions"): {
        "size": 600,
        "overlap": 120,
    },

    ("docs", "tutorials"): {
        "size": 700,
        "overlap": 120,
    },

    ("docs", "user"): {
        "size": 650,
        "overlap": 120,
    },

    ("docs", "solutions"): {
        "size": 700,
        "overlap": 120,
    },

    ("docs", "api"): {
        "size": 500,
        "overlap": 80,
    },

    ("docs", "*"): {
        "size": 650,
        "overlap": 120,
    },

    # --------------------------------------------------------
    # fallback
    # --------------------------------------------------------

    ("*", "*"): {
        "size": 700,
        "overlap": 120,
    },
}

# ============================================================
# HELPERS
# ============================================================

def get_config(source: str, section: str):

    return (
        CHUNK_CONFIG.get((source, section))
        or CHUNK_CONFIG.get((source, "*"))
        or CHUNK_CONFIG[("*", "*")]
    )


def token_count(text: str) -> int:
    return len(ENCODER.encode(text))


def chunk_id(page_id: str, idx: int) -> str:
    return hashlib.sha256(
        f"{page_id}::{idx}".encode()
    ).hexdigest()[:16]

# ============================================================
# MARKDOWN STRUCTURE
# ============================================================

# ONLY H1/H2
#
# Avoids tiny fragmented H3 chunks.

HEADING_RE = re.compile(
    r"^(#{1,2})\s+(.+)$",
    re.MULTILINE
)

TABLE_RE = re.compile(
    r"((?:\|.+\|\n)+)",
    re.MULTILINE
)

LIST_RE = re.compile(
    r"((?:^\s*[-*]\s.+\n?)+)",
    re.MULTILINE
)

# ============================================================
# BLOCK EXTRACTION
# ============================================================

def split_markdown_blocks(text: str):
    """
    Preserve:
      - tables
      - bullet lists
      - paragraphs

    instead of naive blank-line splitting.
    """

    blocks = []

    current = []

    lines = text.splitlines()

    i = 0

    while i < len(lines):

        line = lines[i]

        # ----------------------------------------------------
        # TABLE BLOCK
        # ----------------------------------------------------

        if "|" in line:

            table_lines = [line]

            i += 1

            while i < len(lines):

                if "|" not in lines[i]:
                    break

                table_lines.append(lines[i])

                i += 1

            if current:
                blocks.append(
                    "\n".join(current).strip()
                )
                current = []

            blocks.append(
                "\n".join(table_lines).strip()
            )

            continue

        # ----------------------------------------------------
        # LIST BLOCK
        # ----------------------------------------------------

        if re.match(r"^\s*[-*]\s+", line):

            list_lines = [line]

            i += 1

            while i < len(lines):

                if not re.match(
                    r"^\s*[-*]\s+",
                    lines[i]
                ):
                    break

                list_lines.append(lines[i])

                i += 1

            if current:
                blocks.append(
                    "\n".join(current).strip()
                )
                current = []

            blocks.append(
                "\n".join(list_lines).strip()
            )

            continue

        # ----------------------------------------------------
        # PARAGRAPH BREAK
        # ----------------------------------------------------

        if not line.strip():

            if current:

                blocks.append(
                    "\n".join(current).strip()
                )

                current = []

        else:
            current.append(line)

        i += 1

    if current:

        blocks.append(
            "\n".join(current).strip()
        )

    return [
        b for b in blocks
        if b.strip()
    ]

# ============================================================
# HEADING SEGMENTATION
# ============================================================

def segment_by_headings(text: str):
    """
    Split markdown into:
      (heading, body)

    using ONLY H1/H2.
    """

    lines = text.splitlines()

    segments = []

    current_heading = ""

    current_body = []

    for line in lines:

        if HEADING_RE.match(line):

            body = "\n".join(current_body).strip()

            if body or current_heading:

                segments.append((
                    current_heading,
                    body
                ))

            current_heading = line.strip()

            current_body = []

        else:
            current_body.append(line)

    body = "\n".join(current_body).strip()

    if body or current_heading:

        segments.append((
            current_heading,
            body
        ))

    if not segments:
        return [("", text)]

    return segments

# ============================================================
# OVERLAP
# ============================================================

def paragraph_overlap(
    paragraphs,
    overlap_tokens,
):
    """
    Preserve WHOLE paragraphs
    instead of character tails.
    """

    if not paragraphs:
        return ""

    collected = []

    total = 0

    for para in reversed(paragraphs):

        tokens = token_count(para)

        if total + tokens > overlap_tokens:
            break

        collected.insert(0, para)

        total += tokens

    return "\n\n".join(collected)

# ============================================================
# QUALITY FILTERS
# ============================================================

def is_low_quality(chunk: str):

    stripped = chunk.strip()

    if len(stripped) < 80:
        return True

    # excessive TOC/nav junk
    if stripped.count("](") > 25:
        return True

    # repetitive separators
    if stripped.count("---") > 10:
        return True

    return False

# ============================================================
# CHUNK PACKING
# ============================================================

def pack_blocks(
    blocks,
    chunk_size,
    overlap,
    prefix="",
    allow_sentence_split=True,
):

    chunks = []

    current = []

    current_tokens = token_count(prefix)

    prev_overlap = ""

    def flush():

        nonlocal current
        nonlocal current_tokens
        nonlocal prev_overlap

        if not current:
            return

        body = "\n\n".join(current)

        chunk = body

        # prepend overlap
        if prev_overlap:
            chunk = (
                prev_overlap
                + "\n\n"
                + chunk
            )

        # prepend heading
        if prefix:
            chunk = (
                prefix
                + "\n\n"
                + chunk
            )

        chunk = chunk.strip()

        chunks.append(chunk)

        prev_overlap = paragraph_overlap(
            current,
            overlap,
        )

        current = []

        current_tokens = token_count(prefix)

    # --------------------------------------------------------
    # MAIN BLOCK LOOP
    # --------------------------------------------------------

    for block in blocks:

        block_tokens = token_count(block)

        # fits current chunk
        if (
            current_tokens + block_tokens
            <= chunk_size
        ):

            current.append(block)

            current_tokens += block_tokens

            continue

        # flush existing chunk
        if current:
            flush()

        # ----------------------------------------------------
        # HUGE BLOCK
        # ----------------------------------------------------

        if block_tokens > chunk_size:

            # sales pages:
            # preserve large paragraphs
            if not allow_sentence_split:

                current = [block]

                current_tokens = block_tokens

                flush()

                continue

            # docs pages:
            # sentence fallback allowed
            sentences = re.split(
                r"(?<=[.!?])\s+",
                block,
            )

            sent_buf = []

            sent_tokens = token_count(prefix)

            for sent in sentences:

                stokens = token_count(sent)

                if (
                    sent_tokens + stokens
                    > chunk_size
                    and sent_buf
                ):

                    body = " ".join(sent_buf)

                    chunk = body

                    if prev_overlap:
                        chunk = (
                            prev_overlap
                            + "\n\n"
                            + chunk
                        )

                    if prefix:
                        chunk = (
                            prefix
                            + "\n\n"
                            + chunk
                        )

                    chunks.append(
                        chunk.strip()
                    )

                    prev_overlap = body

                    sent_buf = []

                    sent_tokens = token_count(prefix)

                sent_buf.append(sent)

                sent_tokens += stokens

            if sent_buf:

                current = [
                    " ".join(sent_buf)
                ]

                current_tokens = sent_tokens

        else:

            current = [block]

            current_tokens = (
                token_count(prefix)
                + block_tokens
            )

    flush()

    return chunks

# ============================================================
# PAGE CHUNKING
# ============================================================

def chunk_page(
    text,
    source,
    section,
):

    cfg = get_config(
        source,
        section,
    )

    chunk_size = cfg["size"]

    overlap = cfg["overlap"]

    segments = segment_by_headings(text)

    all_chunks = []

    # sales pages:
    # preserve narratives
    allow_sentence_split = (
        source == "docs"
    )

    for heading, body in segments:

        if not body.strip():
            continue

        blocks = split_markdown_blocks(body)

        if not blocks:
            continue

        chunks = pack_blocks(
            blocks=blocks,
            chunk_size=chunk_size,
            overlap=overlap,
            prefix=heading,
            allow_sentence_split=allow_sentence_split,
        )

        all_chunks.extend(chunks)

    return [
        c for c in all_chunks
        if not is_low_quality(c)
    ]

# ============================================================
# MAIN
# ============================================================

def run():

    if not INPUT_FILE.exists():

        print(
            f"ERROR: {INPUT_FILE} not found"
        )

        return

    total_chunks = 0

    skipped_pages = 0

    stats = {}

    with (
        INPUT_FILE.open(
            encoding="utf-8",
            errors="replace",
        ) as fin,

        OUTPUT_FILE.open(
            "w",
            encoding="utf-8",
        ) as fout
    ):

        for lineno, line in enumerate(fin, 1):

            line = line.strip()

            if not line:
                continue

            try:

                page = json.loads(line)

            except Exception as e:

                print(
                    f"WARN line {lineno}: {e}"
                )

                skipped_pages += 1

                continue

            text = page.get("text", "")

            if not text.strip():

                skipped_pages += 1

                continue

            source = page.get(
                "source",
                "about",
            )

            section = page.get(
                "section",
                "",
            )

            chunks = chunk_page(
                text,
                source,
                section,
            )

            if not chunks:

                skipped_pages += 1

                continue

            for idx, chunk in enumerate(chunks):

                record = {

                    # ----------------------------------------
                    # identity
                    # ----------------------------------------

                    "chunk_id": chunk_id(
                        page["id"],
                        idx,
                    ),

                    "chunk_index": idx,

                    "chunk_total": len(chunks),

                    # ----------------------------------------
                    # text
                    # ----------------------------------------

                    "text": chunk,

                    # ----------------------------------------
                    # metadata
                    # ----------------------------------------

                    "page_id": page["id"],

                    "page_url": page["url"],

                    "page_title": page["title"],

                    "source": source,

                    "section": section,

                    "scraped_at": page["scraped_at"],

                    "word_count": len(
                        chunk.split()
                    ),
                }

                fout.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

                total_chunks += 1

                key = (
                    source,
                    section,
                )

                stats[key] = (
                    stats.get(key, 0) + 1
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)

    print(
        f"Done. "
        f"{total_chunks} chunks written "
        f"-> {OUTPUT_FILE}"
    )

    print(
        f"Skipped pages: {skipped_pages}"
    )

    print("\nChunk stats:\n")

    for key, count in sorted(
        stats.items(),
        key=lambda x: -x[1]
    ):

        source, section = key

        print(
            f"{source:<8} "
            f"{section:<20} "
            f"{count}"
        )

# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    run()