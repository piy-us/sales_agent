"""
agents/query_rewriter_agent.py
────────────────────────────────
Rewrites the raw user query into a rich, standalone search query
suitable for semantic search in Azure AI Search (or any RAG backend).

It uses:
  • conversation history  – to resolve pronouns and anaphora
  • current user query    – as the base
  • (optional) previous failed query + planner feedback – when called in a
    re-write loop because the first retrieval was insufficient

Output is a single optimised search string written to state.rewritten_query.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from config.settings import get_settings
from graph.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a Search Query Rewriter for an enterprise AI assistant.

Your task: transform the user's raw message into the BEST POSSIBLE semantic
search query for a vector search index over enterprise documents.

Rules:
1. Resolve pronouns using conversation history (e.g. "it", "that", "they").
2. Expand abbreviations and acronyms if context makes them clear.
3. Add relevant domain terms the user implied but didn't say.
4. If the planner returned feedback from a previous failed retrieval, 
   incorporate that feedback to find a better angle.
5. Output ONLY the rewritten query string – no explanation, no quotes,
   no JSON wrapper. Just the query text.
"""


def build_query_rewriter_llm() -> AzureChatOpenAI:
    cfg = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=cfg.azure_openai_endpoint,
        api_key=cfg.azure_openai_api_key,
        api_version=cfg.azure_openai_api_version,
        azure_deployment=cfg.azure_chat_deployment,
        temperature=0.1,
        max_tokens=256,
    )


def query_rewriter_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Query Rewriter Agent.

    Reads  : state.user_query, state.conversation_history_text,
             state.planner_feedback (optional, from loop)
    Writes : state.rewritten_query, state.rewrite_count
    """
    rewrite_count = state.get("rewrite_count", 0) + 1
    logger.info("[QueryRewriter] Rewrite attempt #%d", rewrite_count)

    llm = build_query_rewriter_llm()

    history_text = state.get("conversation_history_text", "No prior history.")
    planner_feedback = state.get("planner_feedback", "")

    user_content = (
        f"Conversation history:\n{history_text}\n\n"
        f"User's raw query: {state['user_query']}"
    )
    if planner_feedback:
        user_content += (
            f"\n\nPrevious rewritten query that was insufficient: {state.get('rewritten_query', '')}"
            f"\nPlanner feedback: {planner_feedback}"
            f"\nPlease produce a meaningfully different query to find the missing information."
        )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    rewritten = response.content.strip()

    logger.info("[QueryRewriter] Rewritten query: %r", rewritten)

    return {
        **state,
        "rewritten_query": rewritten,
        "rewrite_count": rewrite_count,
        "next_node": "rag_retrieval",
        "agent_scratchpad": state.get("agent_scratchpad", [])
        + [f"QueryRewriter #{rewrite_count}: {rewritten}"],
    }
