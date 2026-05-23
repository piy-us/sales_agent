"""
graph/state.py
───────────────
Defines the typed state dictionary that flows through the LangGraph graph.

Every node receives the full state and returns a (possibly updated) copy.
TypedDict is used so IDE tooling and mypy can catch field-name typos.
"""
from __future__ import annotations

from typing import Any, TypedDict

from tools.rag_tool import RetrievedChunk


class AgentState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────────
    session_id: str                     # unique conversation identifier
    user_query: str                     # raw text from the user

    # ── Memory ─────────────────────────────────────────────────────────────────
    conversation_history_text: str      # formatted history string for prompts

    # ── Routing ────────────────────────────────────────────────────────────────
    next_node: str                      # "query_rewriter" | "response_writer" | "end"

    # ── Query rewriting ────────────────────────────────────────────────────────
    rewritten_query: str                # optimised search query
    rewrite_count: int                  # number of rewrite attempts so far
    planner_feedback: str               # feedback from context planner on failure

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk]   # chunks from RAG backend

    # ── Images ─────────────────────────────────────────────────────────────────
    fetched_images: list[dict]           # [{"blob_path": str, "base64": str}, ...]

    # ── Output ─────────────────────────────────────────────────────────────────
    final_response: str                 # synthesised answer for the user

    # ── Debug ──────────────────────────────────────────────────────────────────
    agent_scratchpad: list[str]         # trace of agent decisions (for logging)
    error: str                          # error message if pipeline failed
