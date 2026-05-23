"""
agents/context_planner_agent.py
─────────────────────────────────
Evaluates whether the retrieved chunks contain enough relevant information
to answer the user's query, and routes accordingly.

Decision logic
──────────────
• If max relevance score ≥ threshold AND the LLM judges context sufficient
  → route to response_writer
• Otherwise (up to max_rewrite_loops times)
  → route back to query_rewriter with feedback

Future extensions (placeholders present):
  • Web search tool – if RAG insufficient, search the web
  • Image fetcher   – if a chunk references a complex chart/image, fetch it
    from blob storage and pass the base64 bytes to the response writer
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from config.settings import get_settings
from graph.state import AgentState
from tools.rag_tool import RetrievedChunk
from tools.blob_tool import get_blob_fetcher

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a Context Quality Evaluator for an enterprise AI assistant.

You will receive:
  • The user's original question
  • The rewritten search query that was used
  • Retrieved document chunks with relevance scores
  • Conversation history

Your job: decide if the retrieved context is SUFFICIENT to answer the user's question.

Criteria for SUFFICIENT:
  - At least one chunk directly addresses the user's question
  - Enough factual detail to compose a complete, accurate answer
  - The highest relevance score is meaningful (provided separately)

Respond ONLY with valid JSON:
{
  "decision": "sufficient" | "insufficient",
  "feedback": "<if insufficient: specific guidance on what is missing and how to search differently>",
  "image_refs_to_fetch": ["<blob_path>", ...],
  "reasoning": "<one sentence>"
}

For image_refs_to_fetch: only include image blob paths from the chunks if:
  - The image_summary mentions charts, plots, complex diagrams
  - AND those visuals are essential to answer the question
Leave the list empty otherwise.
"""


def build_context_planner_llm() -> AzureChatOpenAI:
    cfg = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=cfg.azure_openai_endpoint,
        api_key=cfg.azure_openai_api_key,
        api_version=cfg.azure_openai_api_version,
        azure_deployment=cfg.azure_chat_deployment,
        temperature=0.1,
        max_tokens=512,
    )


def _format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No chunks retrieved."
    parts = []
    for i, c in enumerate(chunks, 1):
        image_note = ""
        if c.image_refs:
            image_note = f"\n  Image refs: {c.image_refs}"
            if c.image_summary:
                image_note += f"\n  Image summary: {c.image_summary}"
        parts.append(
            f"[Chunk {i}] Source: {c.source} | Score: {c.score:.2f}\n"
            f"  Text: {c.text[:800]}{image_note}"
        )
    return "\n\n".join(parts)


def context_planner_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Context Planner / Quality Gate.

    Reads  : state.retrieved_chunks, state.user_query, state.rewritten_query,
             state.conversation_history_text, state.rewrite_count
    Writes : state.next_node, state.planner_feedback, state.fetched_images
    """
    cfg = get_settings()
    chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])
    rewrite_count = state.get("rewrite_count", 0)

    logger.info(
        "[ContextPlanner] Evaluating %d chunks (rewrite #%d)", len(chunks), rewrite_count
    )

    # Hard cap on loops – avoid infinite cycling
    if rewrite_count >= cfg.max_rewrite_loops:
        logger.warning("[ContextPlanner] Max rewrite loops reached. Proceeding anyway.")
        return {
            **state,
            "next_node": "response_writer",
            "planner_feedback": "",
            "agent_scratchpad": state.get("agent_scratchpad", [])
            + ["ContextPlanner: max loops reached, proceeding with available context"],
        }

    max_score = max((c.score for c in chunks), default=0.0)

    # Quick short-circuit: if score is very low, skip LLM call
    if max_score < 0.3:
        feedback = (
            "All retrieved chunks have very low relevance. "
            "Try different keywords or a broader query angle."
        )
        logger.info("[ContextPlanner] Score too low (%.2f). Routing back.", max_score)
        return {
            **state,
            "next_node": "query_rewriter",
            "planner_feedback": feedback,
            "agent_scratchpad": state.get("agent_scratchpad", [])
            + [f"ContextPlanner: low score {max_score:.2f}, rewriting"],
        }

    # ── LLM-based quality evaluation ───────────────────────────────────────────
    llm = build_context_planner_llm()

    chunks_text = _format_chunks_for_prompt(chunks)
    user_content = (
        f"User question: {state['user_query']}\n"
        f"Rewritten query used: {state.get('rewritten_query', '')}\n"
        f"Max relevance score: {max_score:.2f} (threshold: {cfg.relevance_threshold})\n\n"
        f"Conversation history:\n{state.get('conversation_history_text', 'None')}\n\n"
        f"Retrieved chunks:\n{chunks_text}"
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[ContextPlanner] Could not parse JSON: %s", raw)
        parsed = {"decision": "sufficient", "feedback": "", "image_refs_to_fetch": [], "reasoning": "parse fallback"}

    decision = parsed.get("decision", "sufficient")
    feedback = parsed.get("feedback", "")
    image_refs = parsed.get("image_refs_to_fetch", [])

    logger.info("[ContextPlanner] Decision=%s reason=%s", decision, parsed.get("reasoning"))

    # ── Optionally fetch images (placeholder) ──────────────────────────────────
    fetched_images: list[dict] = []
    if image_refs:
        blob_fetcher = get_blob_fetcher()
        for ref in image_refs:
            try:
                b64 = blob_fetcher.fetch_image_as_base64(ref)
                fetched_images.append({"blob_path": ref, "base64": b64})
                logger.info("[ContextPlanner] Fetched image: %s", ref)
            except Exception as exc:
                logger.error("[ContextPlanner] Failed to fetch %s: %s", ref, exc)

    if decision == "sufficient" or max_score >= cfg.relevance_threshold:
        return {
            **state,
            "next_node": "response_writer",
            "planner_feedback": "",
            "fetched_images": fetched_images,
            "agent_scratchpad": state.get("agent_scratchpad", [])
            + [f"ContextPlanner: context sufficient (score={max_score:.2f})"],
        }
    else:
        return {
            **state,
            "next_node": "query_rewriter",
            "planner_feedback": feedback,
            "fetched_images": [],
            "agent_scratchpad": state.get("agent_scratchpad", [])
            + [f"ContextPlanner: insufficient → feedback: {feedback}"],
        }
