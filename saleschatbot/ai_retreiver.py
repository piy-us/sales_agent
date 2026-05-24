"""
azure_search_retriever.py
============================================================

Azure AI Search — Hybrid Retrieval for Enterprise Sales RAG

Tier compatibility
------------------
FREE tier   → BM25 + vector (RRF fusion)           ✓ fully supported
BASIC tier  → BM25 + vector + semantic reranker     ✓ enable SEMANTIC_RERANKING=true
S1+ tier    → same as Basic + unlimited sem. queries ✓

Retrieval modes
---------------
  hybrid          BM25 + vector, RRF fusion           (default, all tiers)
  vector_only     dense retrieval only
  text_only       BM25 keyword only

Features
--------
  ✓ Top-k with configurable k
  ✓ Semantic reranker (Basic+ only) — toggled via env var
  ✓ Pre-filter by source / section / page_url
  ✓ Returns ranked hits with scores + metadata
  ✓ Convenience wrapper for RAG prompt assembly

INSTALL
-------
pip install azure-search-documents openai

ENV VARS
--------
AZURE_SEARCH_ENDPOINT      https://my-service.search.windows.net
AZURE_SEARCH_ADMIN_KEY     Admin or query key
AZURE_OPENAI_ENDPOINT      (for embedding queries)
AZURE_OPENAI_API_KEY
AZURE_OPENAI_EMBED_MODEL   e.g. text-embedding-3-large
VECTOR_DIMS                3072 (default)
SEMANTIC_RERANKING         false (Free) | true (Basic+)
INDEX_NAME                 enterprise-sales-rag (default)
"""

from __future__ import annotations

import os
import logging

from dataclasses import dataclass, field
from typing import Literal

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
    QueryAnswerType,
    QueryCaptionType,
)

# ============================================================
# CONFIG
# ============================================================

INDEX_NAME         = os.getenv("INDEX_NAME", "enterprise-sales-rag")
VECTOR_DIMS        = int(os.getenv("VECTOR_DIMS", "3072"))
SEMANTIC_RERANKING = os.getenv("SEMANTIC_RERANKING", "false").lower() == "true"
USE_AZURE_OPENAI   = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))

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
# RESULT MODEL
# ============================================================

@dataclass
class SearchResult:
    chunk_id:    str
    text:        str
    page_title:  str
    page_url:    str
    source:      str
    section:     str
    chunk_index: int
    chunk_total: int
    word_count:  int

    # Scores — populated depending on retrieval mode
    search_score:   float = 0.0   # BM25 or RRF hybrid score
    reranker_score: float | None = None  # semantic reranker (Basic+ only)

    # Semantic captions/answers (Basic+ only)
    caption: str | None = None

    def __repr__(self):
        score_str = (
            f"reranker={self.reranker_score:.4f}"
            if self.reranker_score is not None
            else f"rrf={self.search_score:.4f}"
        )
        return (
            f"SearchResult({score_str} | "
            f"{self.source}/{self.section} | "
            f"{self.page_title[:50]!r})"
        )


# ============================================================
# FILTER BUILDER
# ============================================================

def build_filter(
    source: str | None = None,
    section: str | None = None,
    page_url: str | None = None,
    extra_filter: str | None = None,
) -> str | None:
    """
    Build an OData filter string from optional metadata constraints.

    Examples
    --------
    source="about", section="pricing"
        → "source eq 'about' and section eq 'pricing'"

    section in ("pricing", "customers")
        → pass extra_filter="section eq 'pricing' or section eq 'customers'"
    """
    clauses = []

    if source:
        clauses.append(f"source eq '{source}'")

    if section:
        clauses.append(f"section eq '{section}'")

    if page_url:
        clauses.append(f"page_url eq '{page_url}'")

    if extra_filter:
        clauses.append(f"({extra_filter})")

    return " and ".join(clauses) if clauses else None


# ============================================================
# EMBEDDING
# ============================================================

def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    if USE_AZURE_OPENAI:
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


# ============================================================
# SEARCH CLIENT
# ============================================================

def make_search_client() -> SearchClient:
    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
    )


# ============================================================
# CORE RETRIEVER
# ============================================================

def retrieve(
    query: str,
    *,
    top_k: int = 5,
    mode: Literal["hybrid", "vector_only", "text_only"] = "hybrid",
    source: str | None = None,
    section: str | None = None,
    page_url: str | None = None,
    extra_filter: str | None = None,
    semantic_reranking: bool | None = None,
) -> list[SearchResult]:
    """
    Retrieve top-k chunks for a query using hybrid, vector-only, or
    text-only search.

    Parameters
    ----------
    query             : Natural language query string
    top_k             : Number of results to return (default 5)
    mode              : "hybrid" | "vector_only" | "text_only"
    source            : Filter by source ("about" | "docs")
    section           : Filter by section ("pricing", "customers", …)
    page_url          : Filter to a specific page
    extra_filter      : Raw OData filter clause appended with AND
    semantic_reranking: Override the SEMANTIC_RERANKING env var for this call

    Returns
    -------
    List of SearchResult, ordered by relevance (best first).

    Tier notes
    ----------
    FREE  → hybrid (BM25 + HNSW vector + RRF) is fully supported.
            Pass semantic_reranking=False (or leave default).

    BASIC → Enable semantic_reranking=True.
            Azure gives 1 000 free semantic queries / month on Basic;
            set SEMANTIC_RERANKING=true globally or pass it per-call.

    S1+   → Same as Basic but with no monthly query cap.
    """

    use_semantic = (
        semantic_reranking
        if semantic_reranking is not None
        else SEMANTIC_RERANKING
    )

    odata_filter = build_filter(source, section, page_url, extra_filter)

    client = make_search_client()

    # --------------------------------------------------------
    # Build vector query
    # --------------------------------------------------------
    vector_queries = None

    if mode in ("hybrid", "vector_only"):
        query_vector = embed_query(query)
        vector_queries = [
            VectorizedQuery(
                vector=query_vector,
                fields="text_vector",
                # k_nearest_neighbors: how many vector candidates
                # feed into RRF fusion (should be >= top_k)
                k_nearest_neighbors=max(top_k * 3, 50),
                exhaustive=False,   # ANN (HNSW); set True for brute-force
            )
        ]

    # --------------------------------------------------------
    # Build text query (BM25)
    # --------------------------------------------------------
    search_text = query if mode in ("hybrid", "text_only") else None

    # --------------------------------------------------------
    # Fields to retrieve
    # --------------------------------------------------------
    select_fields = [
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

    # --------------------------------------------------------
    # Semantic reranking (Basic+ only)
    # --------------------------------------------------------
    query_type    = None
    query_answer  = None
    query_caption = None
    semantic_cfg  = None

    if use_semantic:
        if mode == "vector_only":
            log.warning(
                "Semantic reranking requires text; "
                "switching to hybrid mode for this call."
            )
            search_text = query
            mode        = "hybrid"

        query_type    = QueryType.SEMANTIC
        semantic_cfg  = "semantic-config"
        query_answer  = QueryAnswerType.EXTRACTIVE
        query_caption = QueryCaptionType.EXTRACTIVE

    # --------------------------------------------------------
    # Execute search
    # --------------------------------------------------------
    log.debug(
        f"search: mode={mode}  top_k={top_k}  "
        f"semantic={use_semantic}  filter={odata_filter!r}"
    )

    raw_results = client.search(
        search_text=search_text,
        vector_queries=vector_queries,
        filter=odata_filter,
        top=top_k,
        select=select_fields,
        query_type=query_type,
        semantic_configuration_name=semantic_cfg,
        query_answer=query_answer,
        query_captions=query_caption,
        # highlight_pre_tag / highlight_post_tag can be added
        # for source-highlighting in a UI
    )

    # --------------------------------------------------------
    # Parse results
    # --------------------------------------------------------
    hits: list[SearchResult] = []

    for r in raw_results:

        caption_text = None
        if hasattr(r, "@search.captions") and r["@search.captions"]:
            caption_text = r["@search.captions"][0].text

        hit = SearchResult(
            chunk_id    = r["chunk_id"],
            text        = r["text"],
            page_title  = r.get("page_title", ""),
            page_url    = r.get("page_url", ""),
            source      = r.get("source", ""),
            section     = r.get("section", ""),
            chunk_index = r.get("chunk_index", 0),
            chunk_total = r.get("chunk_total", 0),
            word_count  = r.get("word_count", 0),
            search_score   = r.get("@search.score", 0.0),
            reranker_score = r.get("@search.rerankerScore"),
            caption        = caption_text,
        )

        hits.append(hit)

    log.info(
        f"Retrieved {len(hits)} results for: {query[:80]!r}"
    )

    return hits


# ============================================================
# RAG CONTEXT ASSEMBLER
# ============================================================

def build_rag_context(
    hits: list[SearchResult],
    *,
    max_tokens: int = 6000,
    include_metadata: bool = True,
) -> str:
    """
    Assemble retrieved chunks into a prompt-ready context block.

    Chunks are included in ranked order until max_tokens is
    approximately reached (estimated at 0.75 tokens/char).

    Parameters
    ----------
    hits           : Output from retrieve()
    max_tokens     : Approximate token budget for context
    include_metadata : Prepend source/title/url per chunk

    Returns
    -------
    A formatted string ready to inject into a system/user prompt.
    """
    parts = []
    total_chars = 0
    char_budget = int(max_tokens / 0.75)

    for i, hit in enumerate(hits, 1):

        if total_chars >= char_budget:
            break

        header = ""
        if include_metadata:
            score_label = (
                f"reranker_score={hit.reranker_score:.4f}"
                if hit.reranker_score is not None
                else f"rrf_score={hit.search_score:.4f}"
            )
            header = (
                f"[{i}] {hit.page_title}\n"
                f"    URL: {hit.page_url}\n"
                f"    Source: {hit.source}/{hit.section}  "
                f"chunk {hit.chunk_index+1}/{hit.chunk_total}  "
                f"{score_label}\n"
            )

        body = hit.caption if hit.caption else hit.text

        chunk_str = f"{header}{body}\n"
        parts.append(chunk_str)
        total_chars += len(chunk_str)

    return "\n---\n".join(parts)


# ============================================================
# CONVENIENCE: SINGLE-CALL RAG FETCH
# ============================================================

def rag_fetch(
    query: str,
    *,
    top_k: int = 5,
    mode: Literal["hybrid", "vector_only", "text_only"] = "hybrid",
    source: str | None = None,
    section: str | None = None,
    semantic_reranking: bool | None = None,
    max_context_tokens: int = 6000,
) -> tuple[str, list[SearchResult]]:
    """
    One-call convenience wrapper.

    Returns
    -------
    (context_string, hits)

    context_string is ready to paste into a prompt.
    hits is the raw list if you need scores / metadata.

    Example
    -------
    context, hits = rag_fetch(
        "What's included in GitLab Ultimate?",
        top_k=6,
        section="pricing",
    )
    prompt = f"Use the context below to answer.\\n\\n{context}\\n\\nQuestion: ..."
    """
    hits = retrieve(
        query,
        top_k=top_k,
        mode=mode,
        source=source,
        section=section,
        semantic_reranking=semantic_reranking,
    )

    context = build_rag_context(hits, max_tokens=max_context_tokens)

    return context, hits


# ============================================================
# QUICK DEMO
# ============================================================

if __name__ == "__main__":

    import textwrap

    QUERIES = [
        # -- pricing / packaging
        "What's included in GitLab Ultimate vs Premium?",
        # -- competitive / enterprise positioning
        "How does GitLab compare to GitHub for enterprise security?",
        # -- customer proof points
        "Enterprise customer success stories with GitLab",
        # -- solution architecture
        "GitLab self-managed vs SaaS deployment options",
    ]

    print("\n" + "=" * 60)
    print("HYBRID SEARCH DEMO")
    tier = "Basic+ (semantic)" if SEMANTIC_RERANKING else "Free (BM25 + vector)"
    print(f"Tier mode : {tier}")
    print(f"Index     : {INDEX_NAME}")
    print("=" * 60)

    for query in QUERIES:

        print(f"\nQ: {query}")
        print("-" * 60)

        hits = retrieve(
            query,
            top_k=3,
            mode="hybrid",
        )

        for i, hit in enumerate(hits, 1):
            score = (
                f"reranker={hit.reranker_score:.4f}"
                if hit.reranker_score is not None
                else f"rrf={hit.search_score:.4f}"
            )
            print(f"  [{i}] {score}  {hit.source}/{hit.section}")
            print(f"       {hit.page_title}")
            print(f"       {hit.page_url}")
            snippet = textwrap.shorten(hit.text, width=120)
            print(f"       {snippet!r}")

    # -- RAG context example
    print("\n\n" + "=" * 60)
    print("RAG CONTEXT EXAMPLE")
    print("=" * 60)

    context, hits = rag_fetch(
        "GitLab pricing for 500 developer seats",
        top_k=4,
        section="pricing",
    )

    print(context[:1200], "…\n")
    print(f"({len(hits)} chunks retrieved)")