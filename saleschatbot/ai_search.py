"""
azure_search_index.py
============================================================

Azure AI Search — Index Creation + Hybrid Search Upload

Designed to ingest output from:
    enterprise_sales_chunker.py  →  chunked_dataset.jsonl

Features:
  ✓ Creates index with hybrid search (BM25 + vector)
  ✓ HNSW vector config for ADA-002 / text-embedding-3-*
  ✓ Semantic ranker config (optional, requires S1+ tier)
  ✓ Batch upload with exponential backoff
  ✓ Embedding generation via Azure OpenAI or OpenAI
  ✓ Progress reporting + error logging
  ✓ Idempotent: safe to re-run (upsert semantics)

INSTALL
-------
pip install azure-search-documents azure-identity openai tqdm

REQUIRED ENV VARS
-----------------
AZURE_SEARCH_ENDPOINT      e.g. https://my-service.search.windows.net
AZURE_SEARCH_ADMIN_KEY     Admin API key (or use managed identity)
AZURE_OPENAI_ENDPOINT      e.g. https://my-oai.openai.azure.com
AZURE_OPENAI_API_KEY       Azure OpenAI key
AZURE_OPENAI_EMBED_MODEL   Deployment name, e.g. text-embedding-3-large

-- OR for plain OpenAI embeddings --
OPENAI_API_KEY             OpenAI key
OPENAI_EMBED_MODEL         e.g. text-embedding-3-large

INPUT
-----
chunked_dataset.jsonl

INDEX
-----
Configurable via INDEX_NAME below.
"""

import json
import os
import time
import logging

from pathlib import Path
from typing import Generator

from tqdm import tqdm

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
    SearchSuggester,
)

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE      = Path("chunked_dataset.jsonl")

INDEX_NAME      = "enterprise-sales-rag"

# Vector dimensions:
#   text-embedding-ada-002       → 1536
#   text-embedding-3-small       → 1536
#   text-embedding-3-large       → 3072
VECTOR_DIMS     = int(os.getenv("VECTOR_DIMS", "3072"))

# Upload batch size (Azure limit: 1000 docs / batch)
BATCH_SIZE      = 100

# Embedding batch size (tokens-per-minute friendly)
EMBED_BATCH     = 16

# Enable semantic ranker (requires Standard S1+ tier)
ENABLE_SEMANTIC = os.getenv("ENABLE_SEMANTIC", "true").lower() == "true"

# Use Azure OpenAI (true) or plain OpenAI (false)
USE_AZURE_OPENAI = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================
# CLIENTS
# ============================================================

def make_search_clients():
    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    key      = os.environ["AZURE_SEARCH_ADMIN_KEY"]
    cred     = AzureKeyCredential(key)

    index_client  = SearchIndexClient(endpoint=endpoint, credential=cred)
    search_client = SearchClient(
        endpoint=endpoint,
        index_name=INDEX_NAME,
        credential=cred,
    )
    return index_client, search_client


def make_embedding_client():
    if USE_AZURE_OPENAI:
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-02-01",
        )
    else:
        from openai import OpenAI
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_embed_model() -> str:
    if USE_AZURE_OPENAI:
        return os.environ["AZURE_OPENAI_EMBED_MODEL"]
    return os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

# ============================================================
# INDEX SCHEMA
# ============================================================

def build_index() -> SearchIndex:
    """
    Hybrid search index schema.

    Filterable fields chosen for enterprise sales RAG:
      - source   (about / docs)
      - section  (pricing / customers / solutions …)
      - page_url (dedup / attribution)
    """

    fields = [

        # ------------------------------------------------
        # Primary key — must be a string in Azure Search
        # ------------------------------------------------
        SimpleField(
            name="chunk_id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),

        # ------------------------------------------------
        # Full-text searchable (BM25 leg of hybrid)
        # ------------------------------------------------
        SearchableField(
            name="text",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),

        SearchableField(
            name="page_title",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),

        # ------------------------------------------------
        # Vector (semantic leg of hybrid)
        # ------------------------------------------------
        SearchField(
            name="text_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMS,
            vector_search_profile_name="hnsw-profile",
        ),

        # ------------------------------------------------
        # Metadata — filterable / facetable
        # ------------------------------------------------
        SimpleField(
            name="page_id",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="page_url",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=False,
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="section",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="chunk_total",
            type=SearchFieldDataType.Int32,
            filterable=False,
        ),
        SimpleField(
            name="word_count",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="scraped_at",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True,
        ),
    ]

    # --------------------------------------------------------
    # VECTOR SEARCH — HNSW
    # --------------------------------------------------------
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                parameters=HnswParameters(
                    m=4,                  # bi-directional links per node
                    ef_construction=400,  # build-time accuracy (higher = slower build, better recall)
                    ef_search=500,        # query-time accuracy
                    metric="cosine",
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-config",
            )
        ],
    )

    # --------------------------------------------------------
    # SEMANTIC RANKER (optional, S1+ tier only)
    # --------------------------------------------------------
    semantic_search = None

    if ENABLE_SEMANTIC:
        semantic_search = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="semantic-config",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="page_title"),
                        content_fields=[
                            SemanticField(field_name="text"),
                        ],
                        keywords_fields=[
                            SemanticField(field_name="section"),
                            SemanticField(field_name="source"),
                        ],
                    ),
                )
            ],
            default_configuration_name="semantic-config",
        )

    return SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

# ============================================================
# EMBEDDING
# ============================================================

def embed_texts(
    client,
    model: str,
    texts: list[str],
    retries: int = 5,
) -> list[list[float]]:
    """
    Batch embed with exponential backoff on rate-limit errors.
    """
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model=model,
                input=texts,
            )
            return [item.embedding for item in response.data]

        except Exception as e:
            wait = 2 ** attempt
            log.warning(
                f"Embedding error (attempt {attempt+1}/{retries}): "
                f"{e}. Retrying in {wait}s…"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Embedding failed after {retries} attempts."
    )

# ============================================================
# DATA LOADING
# ============================================================

def load_chunks(path: Path) -> Generator[dict, None, None]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"Skipping line {lineno}: {e}")


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as f:
        return sum(1 for l in f if l.strip())

# ============================================================
# UPLOAD WITH RETRY
# ============================================================

def upload_batch(
    search_client: SearchClient,
    docs: list[dict],
    retries: int = 5,
):
    for attempt in range(retries):
        try:
            result = search_client.upload_documents(documents=docs)
            failed = [r for r in result if not r.succeeded]
            if failed:
                for r in failed:
                    log.error(
                        f"  Doc {r.key} failed: "
                        f"{r.status_code} {r.error_message}"
                    )
            return
        except Exception as e:
            wait = 2 ** attempt
            log.warning(
                f"Upload error (attempt {attempt+1}/{retries}): "
                f"{e}. Retrying in {wait}s…"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Upload failed after {retries} attempts."
    )

# ============================================================
# MAIN
# ============================================================

def run():

    if not INPUT_FILE.exists():
        log.error(f"{INPUT_FILE} not found. Run enterprise_sales_chunker.py first.")
        return

    # --------------------------------------------------------
    # 1. Create / update index
    # --------------------------------------------------------
    log.info(f"Creating/updating index '{INDEX_NAME}' …")

    index_client, search_client = make_search_clients()
    index_def = build_index()
    index_client.create_or_update_index(index_def)

    log.info("Index ready.")

    # --------------------------------------------------------
    # 2. Set up embedding
    # --------------------------------------------------------
    embed_client = make_embedding_client()
    embed_model  = get_embed_model()
    log.info(f"Embedding model: {embed_model}  dims: {VECTOR_DIMS}")

    # --------------------------------------------------------
    # 3. Stream, embed, upload
    # --------------------------------------------------------
    total_lines   = count_lines(INPUT_FILE)
    total_uploaded = 0
    total_errors  = 0

    embed_buf: list[dict] = []   # accumulate EMBED_BATCH chunks
    upload_buf: list[dict] = []  # accumulate BATCH_SIZE docs

    def flush_embed_buf():
        nonlocal embed_buf

        if not embed_buf:
            return

        texts = [c["text"] for c in embed_buf]

        vectors = embed_texts(embed_client, embed_model, texts)

        for chunk, vec in zip(embed_buf, vectors):
            chunk["text_vector"] = vec

        upload_buf.extend(embed_buf)
        embed_buf = []

    def flush_upload_buf():
        nonlocal upload_buf, total_uploaded

        if not upload_buf:
            return

        upload_batch(search_client, upload_buf)
        total_uploaded += len(upload_buf)
        upload_buf = []

    with tqdm(
        total=total_lines,
        desc="Uploading",
        unit="chunk",
    ) as pbar:

        for chunk in load_chunks(INPUT_FILE):

            # ------------------------------------------------
            # Normalise scraped_at to ISO-8601 with timezone
            # (Azure requires DateTimeOffset to end with Z)
            # ------------------------------------------------
            ts = chunk.get("scraped_at", "")
            if ts and not ts.endswith("Z") and "+" not in ts:
                ts = ts.rstrip("Z") + "Z"
            chunk["scraped_at"] = ts or None

            embed_buf.append(chunk)
            pbar.update(1)

            if len(embed_buf) >= EMBED_BATCH:
                flush_embed_buf()

            if len(upload_buf) >= BATCH_SIZE:
                flush_upload_buf()

        # drain
        flush_embed_buf()
        flush_upload_buf()

    log.info(
        f"\n{'='*60}\n"
        f"Done. {total_uploaded} documents uploaded to '{INDEX_NAME}'.\n"
        f"Errors: {total_errors}\n"
        f"{'='*60}"
    )

# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    run()

# Azure Search
# export AZURE_SEARCH_ENDPOINT="https://your-service.search.windows.net"
# export AZURE_SEARCH_ADMIN_KEY="your-admin-key"

# # Azure OpenAI (or use OPENAI_API_KEY + OPENAI_EMBED_MODEL for plain OpenAI)
# export AZURE_OPENAI_ENDPOINT="https://your-oai.openai.azure.com"
# export AZURE_OPENAI_API_KEY="your-key"
# export AZURE_OPENAI_EMBED_MODEL="text-embedding-3-large"
# export VECTOR_DIMS=3072