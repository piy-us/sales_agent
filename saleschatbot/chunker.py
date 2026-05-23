"""
chunker.py — Takes gitlab_knowledge_base.jsonl and produces chunked_dataset.jsonl
Each chunk is RAG-ready: has its own ID, parent page metadata, and bounded token size.

Run AFTER scraper.py:
    python chunker.py
"""

import json
import hashlib
import re
from pathlib import Path

INPUT_FILE  = Path("gitlab_knowledge_base.jsonl")
OUTPUT_FILE = Path("chunked_dataset.jsonl")

# Tune these for your embedding model:
#   text-embedding-3-small / ada-002 → 512 tokens fine
#   bge-large / e5 → up to 512
#   nomic-embed → up to 8192 (use larger chunks)
CHUNK_SIZE    = 400    # target tokens per chunk (1 token ≈ 4 chars)
CHUNK_OVERLAP = 80     # overlap in tokens to preserve context at boundaries


def token_estimate(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


def split_into_chunks(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Paragraph-aware chunking:
    1. Split on double newlines (paragraph boundaries)
    2. Accumulate paragraphs until chunk_tokens is hit
    3. Slide forward by (chunk_tokens - overlap_tokens)
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    current_paras = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = token_estimate(para)

        # Single paragraph bigger than chunk? Split it by sentences.
        if para_tokens > chunk_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent_tokens = token_estimate(sent)
                if current_tokens + sent_tokens > chunk_tokens and current_paras:
                    chunks.append("\n\n".join(current_paras))
                    # Overlap: keep last few items
                    overlap_text = " ".join(current_paras)[-overlap_tokens * 4:]
                    current_paras = [overlap_text] if overlap_text else []
                    current_tokens = token_estimate(overlap_text)
                current_paras.append(sent)
                current_tokens += sent_tokens
            continue

        if current_tokens + para_tokens > chunk_tokens and current_paras:
            chunks.append("\n\n".join(current_paras))
            # Overlap: retain last N tokens worth of content
            overlap_chars = overlap_tokens * 4
            joined = "\n\n".join(current_paras)
            overlap_text = joined[-overlap_chars:] if len(joined) > overlap_chars else joined
            current_paras = [overlap_text]
            current_tokens = token_estimate(overlap_text)

        current_paras.append(para)
        current_tokens += para_tokens

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return [c for c in chunks if len(c.strip()) > 50]


def chunk_id(page_id: str, idx: int) -> str:
    raw = f"{page_id}::{idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def run():
    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run scraper.py first.")
        return

    total_chunks = 0

    with INPUT_FILE.open(encoding="utf-8") as fin, OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for line in fin:
            page = json.loads(line)
            chunks = split_into_chunks(
                page["text"], CHUNK_SIZE, CHUNK_OVERLAP
            )

            for idx, chunk_text in enumerate(chunks):
                record = {
                    # Chunk identity
                    "chunk_id":     chunk_id(page["id"], idx),
                    "chunk_index":  idx,
                    "chunk_total":  len(chunks),

                    # Text to embed
                    "text": chunk_text,

                    # Parent page metadata (for filtering + attribution)
                    "page_id":      page["id"],
                    "page_url":     page["url"],
                    "page_title":   page["title"],
                    "domain":       page["domain"],
                    "source_type":  page["source_type"],
                    "headings":     page["headings"],
                    "scraped_at":   page["scraped_at"],

                    # For BM25 / hybrid search
                    "word_count":   len(chunk_text.split()),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Done. {total_chunks} chunks → {OUTPUT_FILE}")


if __name__ == "__main__":
    run()