"""
tools/rag_tool.py
──────────────────
PLACEHOLDER – will be replaced by Azure AI Search.

Contract (preserved for the real implementation)
─────────────────────────────────────────────────
retrieve(query: str, top_k: int) → list[RetrievedChunk]

RetrievedChunk fields
    text          : str    – passage text
    source        : str    – document title / URL
    score         : float  – relevance score  0.0–1.0
    image_refs    : list   – blob paths for associated images (optional)
    image_summary : str    – auto-generated caption of the image (optional)

When Azure AI Search is ready:
  1. Replace the body of PlaceholderRAG.retrieve() with a real
     azure.search.documents.SearchClient call.
  2. Map the SearchResult fields to RetrievedChunk.
  3. Everything else in the pipeline stays the same.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float                          # 0.0 – 1.0
    image_refs: list[str] = field(default_factory=list)   # blob paths (future)
    image_summary: str = ""               # LLM-generated caption  (future)


# ── Interface contract ─────────────────────────────────────────────────────────

@runtime_checkable
class RAGBackend(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ...


# ── Placeholder implementation ─────────────────────────────────────────────────

class PlaceholderRAG:
    """
    Returns synthetic results so the graph can run end-to-end
    without a real search backend wired up.

    REPLACE this class body with:
    ────────────────────────────
    from azure.search.documents import SearchClient
    from azure.core.credentials import AzureKeyCredential
    from config.settings import get_settings

    class AzureAISearchRAG:
        def __init__(self):
            cfg = get_settings()
            self._client = SearchClient(
                endpoint=cfg.azure_search_endpoint,
                index_name=cfg.azure_search_index,
                credential=AzureKeyCredential(cfg.azure_search_key),
            )

        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
            results = self._client.search(
                search_text=query,
                top=top_k,
                query_type="semantic",
                semantic_configuration_name="default",
                select=["content", "source", "@search.rerankerScore"],
            )
            chunks = []
            for r in results:
                chunks.append(RetrievedChunk(
                    text=r["content"],
                    source=r["source"],
                    score=r.get("@search.rerankerScore", 0.5) / 4.0,  # normalise 0-4 → 0-1
                    image_refs=r.get("image_refs", []),
                    image_summary=r.get("image_summary", ""),
                ))
            return chunks
    """

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        logger.info("[PLACEHOLDER RAG] query=%r top_k=%d", query, top_k)
        # Return stubbed results with a mid-range score
        return [
            RetrievedChunk(
                text=(
                    f"[PLACEHOLDER] This is a simulated retrieved passage for the "
                    f"query: '{query}'. "
                    "Replace PlaceholderRAG with AzureAISearchRAG (see docstring) "
                    "to get real enterprise data."
                ),
                source="placeholder://enterprise-docs/doc1",
                score=0.72,   # above default threshold → context planner will proceed
                image_refs=[],
                image_summary="",
            )
        ]


# ── Singleton factory ──────────────────────────────────────────────────────────

_rag_instance: RAGBackend | None = None


def get_rag_backend() -> RAGBackend:
    """
    Return the active RAG backend.
    Swap PlaceholderRAG → AzureAISearchRAG here when ready.
    """
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = PlaceholderRAG()
    return _rag_instance
