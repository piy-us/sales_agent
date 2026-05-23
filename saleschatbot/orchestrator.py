"""
graph/orchestrator.py
──────────────────────
Defines and compiles the LangGraph StateGraph that wires all agents together.

Graph topology
──────────────
                      ┌─────────────────────────────────────┐
                      │                                     │
  START → [conversation_agent]                             │
               │                                           │
         ┌─────┴──────┐                                    │
         │ direct     │ rag                                │
         ▼            ▼                                    │
        END  [query_rewriter_agent]                        │
                      │                                    │
                      ▼                                    │
              [rag_retrieval_node]                         │
                      │                                    │
                      ▼                                    │
           [context_planner_agent]                         │
                      │                                    │
            ┌─────────┴──────────┐                        │
            │ sufficient         │ insufficient            │
            ▼                   └────────────────────────-┘
    [response_writer_agent]
            │
           END
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from graph.state import AgentState
from agents.conversation_agent import conversation_agent_node
from agents.query_rewriter_agent import query_rewriter_node
from agents.context_planner_agent import context_planner_node
from agents.response_writer_agent import response_writer_node
from tools.rag_tool import get_rag_backend

logger = logging.getLogger(__name__)


# ── RAG retrieval node (thin wrapper around the backend) ───────────────────────

def rag_retrieval_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: calls the RAG backend with the rewritten query.

    When Azure AI Search is integrated, PlaceholderRAG → AzureAISearchRAG
    automatically via get_rag_backend().
    """
    query = state.get("rewritten_query") or state.get("user_query", "")
    logger.info("[RAGRetrieval] Searching for: %r", query)

    backend = get_rag_backend()
    chunks = backend.retrieve(query=query, top_k=5)

    logger.info("[RAGRetrieval] Retrieved %d chunks", len(chunks))

    return {
        **state,
        "retrieved_chunks": chunks,
        "next_node": "context_planner",
        "agent_scratchpad": state.get("agent_scratchpad", [])
        + [f"RAGRetrieval: {len(chunks)} chunks for query={query!r}"],
    }


# ── Conditional edge functions ─────────────────────────────────────────────────

def route_after_conversation(state: AgentState) -> str:
    """Route after the conversation agent based on next_node."""
    nxt = state.get("next_node", "end")
    logger.debug("[Router:conversation] → %s", nxt)
    if nxt == "query_rewriter":
        return "query_rewriter"
    return END


def route_after_context_planner(state: AgentState) -> str:
    """Route after the context planner: loop or proceed."""
    nxt = state.get("next_node", "response_writer")
    logger.debug("[Router:context_planner] → %s", nxt)
    if nxt == "query_rewriter":
        return "query_rewriter"
    return "response_writer"


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph.
    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    graph.add_node("conversation_agent", conversation_agent_node)
    graph.add_node("query_rewriter", query_rewriter_node)
    graph.add_node("rag_retrieval", rag_retrieval_node)
    graph.add_node("context_planner", context_planner_node)
    graph.add_node("response_writer", response_writer_node)

    # ── Entry point ────────────────────────────────────────────────────────────
    graph.add_edge(START, "conversation_agent")

    # ── Conditional: conversation agent → end OR query rewriter ───────────────
    graph.add_conditional_edges(
        "conversation_agent",
        route_after_conversation,
        {
            "query_rewriter": "query_rewriter",
            END: END,
        },
    )

    # ── Linear: query rewriter → RAG retrieval ────────────────────────────────
    graph.add_edge("query_rewriter", "rag_retrieval")

    # ── Linear: RAG retrieval → context planner ───────────────────────────────
    graph.add_edge("rag_retrieval", "context_planner")

    # ── Conditional: context planner → response writer OR back to query rewriter
    graph.add_conditional_edges(
        "context_planner",
        route_after_context_planner,
        {
            "response_writer": "response_writer",
            "query_rewriter": "query_rewriter",
        },
    )

    # ── Linear: response writer → END ─────────────────────────────────────────
    graph.add_edge("response_writer", END)

    compiled = graph.compile()
    logger.info("[Orchestrator] Graph compiled successfully")
    return compiled


# ── Module-level singleton ─────────────────────────────────────────────────────

_compiled_graph = None


def get_graph() -> Any:
    """Return the compiled graph singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
