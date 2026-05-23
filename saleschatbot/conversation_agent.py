"""
agents/conversation_agent.py
─────────────────────────────
The entry-point agent.  It has two responsibilities:

1. Direct response  – handle greetings, general chitchat, capability
   explanations, short factual questions that do NOT need enterprise data.

2. Route to RAG pipeline – for anything that needs real enterprise data,
   set next_node = "query_rewriter".

The agent uses an LLM with a structured output schema so the decision is
machine-readable rather than parsed from free text.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from config.settings import get_settings
from graph.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Conversation Router for an enterprise AI assistant.

Your job is to decide ONE of two things for every user message:

A) DIRECT – you can answer the message yourself RIGHT NOW.
   This includes: greetings, farewells, capability questions ("what can you do?"),
   simple clarifications, small talk, or any question that does NOT require
   searching internal enterprise documents.

B) RAG – the user is asking something that needs information from the
   enterprise knowledge base. Forward the query to the RAG pipeline.

You MUST respond ONLY with a valid JSON object in this exact schema:
{
  "decision": "direct" | "rag",
  "direct_response": "<your reply if decision=direct, else empty string>",
  "reasoning": "<one sentence explaining your choice>"
}

No markdown, no explanation outside the JSON.
"""


def build_conversation_agent() -> AzureChatOpenAI:
    cfg = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=cfg.azure_openai_endpoint,
        api_key=cfg.azure_openai_api_key,
        api_version=cfg.azure_openai_api_version,
        azure_deployment=cfg.azure_chat_deployment,
        temperature=0.2,
        max_tokens=512,
    )


def conversation_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Conversation / Router Agent.

    Reads  : state.user_query, state.conversation_history
    Writes : state.next_node, state.final_response (if direct)
    """
    logger.info("[ConversationAgent] Processing query: %r", state["user_query"])

    llm = build_conversation_agent()

    history_text = state.get("conversation_history_text", "No prior history.")
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Conversation history so far:\n{history_text}\n\n"
                f"Current user message: {state['user_query']}"
            )
        ),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[ConversationAgent] Failed to parse JSON: %s", raw)
        # Fallback: treat as direct response
        parsed = {
            "decision": "direct",
            "direct_response": raw,
            "reasoning": "JSON parse fallback",
        }

    decision = parsed.get("decision", "direct")
    logger.info("[ConversationAgent] Decision=%s reason=%s", decision, parsed.get("reasoning"))

    if decision == "rag":
        return {
            **state,
            "next_node": "query_rewriter",
            "agent_scratchpad": state.get("agent_scratchpad", [])
            + [f"ConversationAgent → routed to RAG: {parsed.get('reasoning')}"],
        }
    else:
        return {
            **state,
            "next_node": "end",
            "final_response": parsed.get("direct_response", ""),
            "agent_scratchpad": state.get("agent_scratchpad", [])
            + [f"ConversationAgent → direct reply: {parsed.get('reasoning')}"],
        }
