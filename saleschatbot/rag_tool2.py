"""
tools/rag_tool.py
──────────────────
Azure AI Search hybrid RAG backend for the LangGraph orchestration.

Pipeline position
─────────────────
  query_rewriter_agent
        │  state.rewritten_query
        ▼
  rag_retrieval_node  (orchestrator.py)
        │  calls backend.retrieve(query, top_k)
        ▼
  [this file]  AzureAISearchRAG.retrieve()
        │  returns list[RetrievedChunk]
        ▼
  context_planner_agent
        │  state.retrieved_chunks
        ▼
  response_writer_agent

RetrievedChunk contract (unchanged — all downstream agents depend on this)
────────────────────────────────────────────────────────────────────────────
  text          : str    – passage text
  source        : str    – page_url  (was doc title in placeholder)
  score         : float  – relevance score normalised to 0.0–1.0
  image_refs    : list   – blob paths for associated images (kept for future)
  image_summary : str    – image caption (kept for future)

Score normalisation
────────────────────
  Hybrid RRF scores from Azure AI Search are small floats, typically
  0.01–0.10 range, NOT 0–1.  We normalise within each result batch so
  context_planner_agent's threshold checks (score >= 0.3 etc.) still work
  without changing any of that agent's logic.

  Formula: normalised = score / max_score_in_batch   (safe, avoids div/0)
  If all scores are 0, everything maps to 0.0.

ENV VARS (same as azure_search_retriever.py)
────────────────────────────────────────────
  AZURE_SEARCH_ENDPOINT      https://my-service.search.windows.net
  AZURE_SEARCH_ADMIN_KEY     admin or query key
  AZURE_OPENAI_ENDPOINT      for embedding the query
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_EMBED_MODEL   e.g. text-embedding-3-large
  VECTOR_DIMS                3072 (default)
  INDEX_NAME                 enterprise-sales-rag (default)

  SEMANTIC_RERANKING is intentionally NOT used here — we are on Free tier,
  doing hybrid (BM25 + vector RRF) only.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

logger = logging.getLogger(__name__)


# ── Data model (unchanged contract) ───────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float                                        # normalised 0.0–1.0
    image_refs: list[str] = field(default_factory=list) # reserved for future
    image_summary: str = ""                             # reserved for future

    # ── Extra fields surfaced from Azure AI Search ─────────────────────────
    # These are NOT used by existing agents but are available for future use
    # (e.g. response_writer citing page_title / chunk navigation).
    chunk_id:    str = ""
    page_title:  str = ""
    section:     str = ""
    chunk_index: int = 0
    chunk_total: int = 0


# ── Interface contract ─────────────────────────────────────────────────────────

@runtime_checkable
class RAGBackend(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ...


# ── Embedding helper ───────────────────────────────────────────────────────────

def _embed_query(text: str) -> list[float]:
    """Embed a single query string using Azure OpenAI or plain OpenAI."""
    use_azure = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))

    if use_azure:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-02-01",
        )
        model = os.environ["AZURE_OPENAI_EMBED_MODEL"]
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

    response = client.embeddings.create(model=model, input=[text])
    return response.data[0].embedding


# ── Score normalisation ────────────────────────────────────────────────────────

def _normalise_scores(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Normalise raw Azure RRF scores (typically 0.01–0.10) to 0.0–1.0
    so context_planner_agent threshold checks work without modification.

    Uses min-max normalisation within the batch:
        normalised = (score - min) / (max - min)

    If all scores are equal (including all-zero), maps everything to 1.0
    (treat as equally relevant, let the planner decide).
    """
    if not chunks:
        return chunks

    scores = [c.score for c in chunks]
    min_s, max_s = min(scores), max(scores)

    if max_s == min_s:
        # All equal — assign 1.0 so planner doesn't reject them outright
        for c in chunks:
            c.score = 1.0
        return chunks

    for c in chunks:
        c.score = (c.score - min_s) / (max_s - min_s)

    return chunks


# ── Azure AI Search backend ────────────────────────────────────────────────────

class AzureAISearchRAG:
    """
    Hybrid (BM25 + vector, RRF fusion) RAG backend backed by Azure AI Search.

    No semantic reranking — compatible with Free tier.
    Drop-in replacement for PlaceholderRAG; same retrieve() signature.
    """

    # Fields to pull back from the index (matches azure_search_index.py schema)
    _SELECT_FIELDS = [
        "chunk_id",
        "text",
        "page_title",
        "page_url",
        "source",
        "section",
        "chunk_index",
        "chunk_total",
        "word_count",
    ]

    def __init__(self):
        endpoint  = os.environ["AZURE_SEARCH_ENDPOINT"]
        key       = os.environ["AZURE_SEARCH_ADMIN_KEY"]
        index     = os.getenv("INDEX_NAME", "enterprise-sales-rag")
        self._vector_dims = int(os.getenv("VECTOR_DIMS", "3072"))

        self._client = SearchClient(
            endpoint=endpoint,
            index_name=index,
            credential=AzureKeyCredential(key),
        )
        logger.info(
            "[AzureAISearchRAG] Connected → endpoint=%s  index=%s",
            endpoint, index,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        source_filter: str | None = None,
        section_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Hybrid retrieval: BM25 + HNSW vector, fused with RRF.

        Parameters
        ----------
        query          : Rewritten query from query_rewriter_agent
        top_k          : Number of results (orchestrator passes 5 by default)
        source_filter  : Optional OData filter on the 'source' field
                         e.g. "about" or "docs"
        section_filter : Optional OData filter on the 'section' field
                         e.g. "pricing" or "customers"

        Returns
        -------
        list[RetrievedChunk] — scores normalised to 0.0–1.0, best first
        """
        logger.info(
            "[AzureAISearchRAG] query=%r  top_k=%d", query, top_k
        )

        # ── Embed the query ────────────────────────────────────────────────
        query_vector = _embed_query(query)

        vector_query = VectorizedQuery(
            vector=query_vector,
            fields="text_vector",
            # Feed more vector candidates into RRF than top_k to avoid
            # recall loss at small k values
            k_nearest_neighbors=max(top_k * 4, 50),
            exhaustive=False,  # ANN (HNSW) — fast, Free-tier safe
        )

        # ── Build optional OData pre-filter ───────────────────────────────
        filter_clauses = []
        if source_filter:
            filter_clauses.append(f"source eq '{source_filter}'")
        if section_filter:
            filter_clauses.append(f"section eq '{section_filter}'")
        odata_filter = " and ".join(filter_clauses) or None

        # ── Execute hybrid search ──────────────────────────────────────────
        raw_results = self._client.search(
            search_text=query,          # BM25 leg
            vector_queries=[vector_query],  # vector leg
            filter=odata_filter,
            top=top_k,
            select=self._SELECT_FIELDS,
            # No query_type="semantic" — Free tier, hybrid RRF only
        )

        # ── Map Azure results → RetrievedChunk ────────────────────────────
        chunks: list[RetrievedChunk] = []

        for r in raw_results:
            raw_score = r.get("@search.score", 0.0)

            chunk = RetrievedChunk(
                # ── Core contract fields (used by existing agents) ─────────
                text   = r.get("text", ""),
                source = r.get("page_url", r.get("source", "")),
                score  = raw_score,   # will be normalised below

                # ── Extended fields (available for future agent use) ───────
                chunk_id    = r.get("chunk_id", ""),
                page_title  = r.get("page_title", ""),
                section     = r.get("section", ""),
                chunk_index = r.get("chunk_index", 0),
                chunk_total = r.get("chunk_total", 0),
            )
            chunks.append(chunk)

        logger.info(
            "[AzureAISearchRAG] %d chunks retrieved, raw scores: %s",
            len(chunks),
            [round(c.score, 4) for c in chunks],
        )

        # Normalise scores so downstream threshold checks work correctly
        chunks = _normalise_scores(chunks)

        logger.info(
            "[AzureAISearchRAG] normalised scores: %s",
            [round(c.score, 4) for c in chunks],
        )

        return chunks


# ── Placeholder (kept for local runs without Azure credentials) ───────────────

class PlaceholderRAG:
    """
    Synthetic fallback for local development without Azure credentials.
    Set RAG_BACKEND=placeholder to force this.
    """

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        logger.warning(
            "[PlaceholderRAG] Using stub results — set Azure env vars to use real search."
        )
        return [
            RetrievedChunk(
                text=(
                    f"[PLACEHOLDER] Simulated passage for query: '{query}'. "
                    "Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_ADMIN_KEY "
                    "to enable real hybrid search."
                ),
                source="placeholder://enterprise-docs/doc1",
                score=0.72,
            )
        ]


# ── Singleton factory ──────────────────────────────────────────────────────────

_rag_instance: RAGBackend | None = None


def get_rag_backend() -> RAGBackend:
    """
    Return the active RAG backend singleton.

    Selection logic:
      RAG_BACKEND=placeholder  → PlaceholderRAG  (local dev, no Azure needed)
      AZURE_SEARCH_ENDPOINT set → AzureAISearchRAG  (default when env is set)
      fallback                 → PlaceholderRAG
    """
    global _rag_instance

    if _rag_instance is None:
        backend_choice = os.getenv("RAG_BACKEND", "").lower()

        if backend_choice == "placeholder":
            logger.info("[RAGFactory] Using PlaceholderRAG (forced by RAG_BACKEND env)")
            _rag_instance = PlaceholderRAG()

        elif os.getenv("AZURE_SEARCH_ENDPOINT"):
            logger.info("[RAGFactory] Using AzureAISearchRAG")
            _rag_instance = AzureAISearchRAG()

        else:
            logger.warning(
                "[RAGFactory] AZURE_SEARCH_ENDPOINT not set — "
                "falling back to PlaceholderRAG. "
                "Set it (and AZURE_SEARCH_ADMIN_KEY) to use real search."
            )
            _rag_instance = PlaceholderRAG()

    return _rag_instance