"""
uploader.py
─────────────────────────────────────────────────────────────────────────────
STEP 3  Upload slide images to Azure Blob Storage.
STEP 5  Embed chunks with Azure OpenAI text-embedding model (via Foundry
        endpoint) and index them into Azure AI Search.

pip install azure-storage-blob azure-search-documents openai
"""

from __future__ import annotations

import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.storage.blob import BlobServiceClient, ContentSettings
from openai import AzureOpenAI

# ── Config ─────────────────────────────────────────────────────────────────
AZURE_SEARCH_ENDPOINT  = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_KEY       = os.environ["AZURE_SEARCH_KEY"]
AZURE_STORAGE_CONN     = os.environ["AZURE_STORAGE_CONN"]

# Azure OpenAI / Foundry
AZURE_FOUNDRY_ENDPOINT  = os.environ.get(
    "AZURE_FOUNDRY_ENDPOINT", "https://services.ai.azure.com/v1"
)
AZURE_OPENAI_KEY        = os.environ["AZURE_OPENAI_KEY"]
AZURE_OPENAI_API_VER    = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
EMBED_DEPLOYMENT        = os.environ.get(
    "AZURE_EMBED_DEPLOYMENT", "text-embedding-3-large"
)
EMBED_DIMENSIONS        = int(os.environ.get("AZURE_EMBED_DIMENSIONS", "1536"))

INDEX_NAME      = os.environ.get("AZURE_SEARCH_INDEX", "pptx-rag-index")
BLOB_CONTAINER  = os.environ.get("AZURE_BLOB_CONTAINER", "slide-images")

# ── Lazy singletons ────────────────────────────────────────────────────────
_embed_client:   AzureOpenAI | None       = None
_blob_service:   BlobServiceClient | None = None


def _get_embed_client() -> AzureOpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = AzureOpenAI(
            azure_endpoint = AZURE_FOUNDRY_ENDPOINT,
            api_key        = AZURE_OPENAI_KEY,
            api_version    = AZURE_OPENAI_API_VER,
        )
    return _embed_client


def _get_blob_service() -> BlobServiceClient:
    global _blob_service
    if _blob_service is None:
        _blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONN)
    return _blob_service


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — Blob Storage upload
# ═══════════════════════════════════════════════════════════════════

def upload_image_to_blob(blob_bytes: bytes, filename: str) -> str:
    """
    Upload raw image bytes to Azure Blob Storage.

    Returns
    -------
    str  URL of the uploaded blob (use a SAS URL in production).
    """
    service          = _get_blob_service()
    container_client = service.get_container_client(BLOB_CONTAINER)

    try:
        container_client.create_container()
    except Exception:
        pass   # container already exists

    blob_client = container_client.get_blob_client(filename)
    blob_client.upload_blob(
        blob_bytes,
        overwrite        = True,
        content_settings = ContentSettings(content_type="image/png"),
    )
    print(f"    ☁️  Uploaded blob: {filename}")
    return blob_client.url   # swap for SAS URL in production


# ═══════════════════════════════════════════════════════════════════
# STEP 5a — Embedding via Azure OpenAI (Foundry endpoint)
# ═══════════════════════════════════════════════════════════════════

def get_embedding(text: str) -> list[float]:
    """
    Embed *text* using the Azure OpenAI embedding deployment configured in
    AZURE_EMBED_DEPLOYMENT, called through the AI Foundry endpoint.
    """
    client   = _get_embed_client()
    response = client.embeddings.create(
        input      = text,
        model      = EMBED_DEPLOYMENT,
        dimensions = EMBED_DIMENSIONS,   # only supported by v3 models
    )
    return response.data[0].embedding


# ═══════════════════════════════════════════════════════════════════
# STEP 5b — Azure AI Search index management
# ═══════════════════════════════════════════════════════════════════

def create_index_if_missing() -> None:
    """Create the Azure AI Search index if it does not exist yet."""
    index_client = SearchIndexClient(
        endpoint   = AZURE_SEARCH_ENDPOINT,
        credential = AzureKeyCredential(AZURE_SEARCH_KEY),
    )

    fields = [
        SimpleField(
            name="id", type=SearchFieldDataType.String, key=True
        ),
        SearchableField(
            name="text", type=SearchFieldDataType.String
        ),
        SimpleField(
            name="content_type",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="source_file",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="slide_number",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="slide_number_end",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="slide_title", type=SearchFieldDataType.String
        ),
        SimpleField(
            name="image_uri", type=SearchFieldDataType.String
        ),
        SearchField(
            name                       = "embedding",
            type                       = SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),
            searchable                 = True,
            vector_search_dimensions   = EMBED_DIMENSIONS,
            vector_search_profile_name = "hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms = [HnswAlgorithmConfiguration(name="hnsw")],
        profiles   = [
            VectorSearchProfile(
                name                       = "hnsw-profile",
                algorithm_configuration_name = "hnsw",
            )
        ],
    )

    index = SearchIndex(
        name          = INDEX_NAME,
        fields        = fields,
        vector_search = vector_search,
    )
    index_client.create_or_update_index(index)
    print(f"  ✅ Index '{INDEX_NAME}' ready")


# ═══════════════════════════════════════════════════════════════════
# STEP 5c — Index chunks
# ═══════════════════════════════════════════════════════════════════

def index_chunks(chunks: list[dict], batch_size: int = 100) -> None:
    """
    Embed each chunk and upload to Azure AI Search in batches.

    Parameters
    ----------
    chunks      : list of dicts produced by parser.assemble_chunks()
    batch_size  : number of documents per upload batch (max 1000 for SDK)
    """
    create_index_if_missing()

    search_client = SearchClient(
        endpoint   = AZURE_SEARCH_ENDPOINT,
        index_name = INDEX_NAME,
        credential = AzureKeyCredential(AZURE_SEARCH_KEY),
    )

    documents: list[dict] = []
    for chunk in chunks:
        print(f"  📐 Embedding chunk: {chunk['id'][:20]}…")
        meta = chunk.get("metadata", {})
        doc  = {
            "id":              chunk["id"],
            "text":            chunk["text"],
            "content_type":    chunk["content_type"],
            "source_file":     meta.get("source_file", ""),
            "slide_number":    meta.get("slide_number", 0),
            "slide_number_end": meta.get("slide_number_end",
                                         meta.get("slide_number", 0)),
            "slide_title":     meta.get("slide_title", ""),
            "image_uri":       chunk.get("image_uri") or "",
            "embedding":       get_embedding(chunk["text"]),
        }
        documents.append(doc)

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        search_client.upload_documents(batch)
        print(f"  ✅ Uploaded batch {i // batch_size + 1} "
              f"({len(batch)} docs)")

    print(f"\n  ✅ Indexed {len(documents)} chunks total")
