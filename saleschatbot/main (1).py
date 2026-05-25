"""
main.py
─────────────────────────────────────────────────────────────────────────────
Orchestrator — glues parser → uploader → retriever together.

Usage
-----
# Ingest a deck
python main.py ingest path/to/deck.pptx

# Query (optional: pass a question as positional args)
python main.py query "What does the revenue chart show for Q3?"

Required environment variables
───────────────────────────────
AZURE_SEARCH_ENDPOINT   e.g. https://my-search.search.windows.net
AZURE_SEARCH_KEY        admin key for the search service
AZURE_STORAGE_CONN      connection string for blob storage
AZURE_OPENAI_KEY        key for the Azure OpenAI / AI Foundry resource

Optional environment variables (sane defaults provided)
────────────────────────────────────────────────────────
AZURE_FOUNDRY_ENDPOINT          default: https://services.ai.azure.com/v1
AZURE_OPENAI_API_VERSION        default: 2025-01-01-preview
AZURE_VISION_DEPLOYMENT         default: gpt-4o
AZURE_CHAT_DEPLOYMENT           default: gpt-4o
AZURE_EMBED_DEPLOYMENT          default: text-embedding-3-large
AZURE_EMBED_DIMENSIONS          default: 1536
AZURE_SEARCH_INDEX              default: pptx-rag-index
AZURE_BLOB_CONTAINER            default: slide-images
"""

import sys

from parser    import parse_pptx, assemble_chunks
from uploader  import upload_image_to_blob, index_chunks
from retriever import answer_query


def ingest(pptx_path: str) -> None:
    print("\n=== STEP 1 + 2: Parse & describe visuals ===")
    parsed = parse_pptx(pptx_path)
    print(f"  Parsed {parsed['file_metadata']['slide_count']} slides")

    print("\n=== STEP 3 + 4: Upload images & assemble overlapping chunks ===")
    chunks = assemble_chunks(parsed, upload_fn=upload_image_to_blob)
    print(f"  Assembled {len(chunks)} chunks "
          f"(window={4}, overlap={2}, stride={2})")

    print("\n=== STEP 5: Embed & index ===")
    index_chunks(chunks)
    print("\n✅ Ingestion complete.")


def query(question: str) -> None:
    print(f"\n❓ Question: {question}\n")
    answer = answer_query(question)
    print(f"💬 Answer:\n{answer}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python main.py ingest <path/to/deck.pptx>")
            sys.exit(1)
        ingest(sys.argv[2])

    elif mode == "query":
        q = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else \
            "What does the revenue chart show for Q3?"
        query(q)

    else:
        print(f"Unknown mode '{mode}'. Use 'ingest' or 'query'.")
        sys.exit(1)
