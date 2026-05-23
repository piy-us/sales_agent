"""
chunker.py — RAG-ready chunking for the GitLab sales knowledge base
====================================================================
Reads:  gitlab_kb.jsonl         (from scraper.py)
Writes: chunked_dataset.jsonl   (embed-ready, one chunk per line)

Chunking strategy:
- Paragraph-aware: never splits mid-paragraph
- Overlap: last N tokens of previous chunk repeated at start of next
  (prevents answers being cut at chunk boundaries)
- Priority boost: HIGH relevance pages get smaller chunks (more precise retrieval)
"""

import json, hashlib, re
from pathlib import Path

INPUT_FILE  = Path("gitlab_knowledge_base.jsonl")
OUTPUT_FILE = Path("chunked_datasetabout.jsonl")

# Token budgets by relevance tier
# Smaller = more precise retrieval (good for pricing/solutions)
# Larger  = more context per chunk (good for topics/explanations)
CHUNK_CONFIG = {
    "HIGH":   {"size": 350, "overlap": 70},
    "MEDIUM": {"size": 450, "overlap": 80},
    "LOW":    {"size": 500, "overlap": 80},
}

def token_est(text: str) -> int:
    return len(text) // 4   # 1 token ≈ 4 chars (conservative)

def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """
    Paragraph-aware chunker with sliding overlap window.
    Splits on double newlines first, then sentences if a paragraph is huge.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current, current_tokens = [], [], 0

    for para in paragraphs:
        ptokens = token_est(para)

        # Single paragraph larger than chunk? Split by sentences.
        if ptokens > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                stokens = token_est(sent)
                if current_tokens + stokens > chunk_size and current:
                    chunks.append("\n\n".join(current))
                    # Overlap: keep tail of previous chunk
                    tail = " ".join(current)[-overlap * 4:]
                    current = [tail] if tail else []
                    current_tokens = token_est(tail)
                current.append(sent)
                current_tokens += stokens
            continue

        if current_tokens + ptokens > chunk_size and current:
            chunks.append("\n\n".join(current))
            tail = "\n\n".join(current)[-overlap * 4:]
            current = [tail] if tail else []
            current_tokens = token_est(tail)

        current.append(para)
        current_tokens += ptokens

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if len(c.strip()) > 60]


def chunk_id(page_id: str, idx: int) -> str:
    return hashlib.sha256(f"{page_id}::{idx}".encode()).hexdigest()[:16]


def run():
    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run scraper.py first.")
        return

    total = 0
    stats = {}  # category → chunk count

    with INPUT_FILE.open() as fin, OUTPUT_FILE.open("w") as fout:
        for line in fin:
            page = json.loads(line)
            cfg  = CHUNK_CONFIG.get(page.get("relevance", "MEDIUM"), CHUNK_CONFIG["MEDIUM"])
            chunks = chunk_text(page["text"], cfg["size"], cfg["overlap"])

            for idx, chunk in enumerate(chunks):
                record = {
                    # ── Chunk identity ─────────────────────────────────
                    "chunk_id":     chunk_id(page["id"], idx),
                    "chunk_index":  idx,
                    "chunk_total":  len(chunks),

                    # ── Text to embed ──────────────────────────────────
                    "text": chunk,

                    # ── Rich metadata for filtered RAG retrieval ───────
                    # Use these in your vector DB metadata filters:
                    #   e.g. filter(category="pricing") for pricing questions
                    #        filter(relevance="HIGH") to boost precision
                    "page_id":      page["id"],
                    "page_url":     page["url"],
                    "page_title":   page["title"],
                    "category":     page["category"],
                    "relevance":    page["relevance"],
                    "headings":     page["headings"],
                    "scraped_at":   page["scraped_at"],
                    "word_count":   len(chunk.split()),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
                stats[page["category"]] = stats.get(page["category"], 0) + 1

    print(f"\nDone. {total} chunks → {OUTPUT_FILE}")
    print("\nChunks by category:")
    for cat, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {cat:<15} {count}")


if __name__ == "__main__":
    run()