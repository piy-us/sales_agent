"""
agents/response_writer_agent.py
─────────────────────────────────
Synthesises the final answer from:
  • retrieved document chunks
  • fetched images (base64, optional – from context planner)
  • conversation history
  • original user query

Uses a VISION-capable model (gpt-4.1 / gpt-4o) so it can reason over
both text and images in the same call.

The agent produces a well-structured markdown response ready for the user.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from config.settings import get_settings
from graph.state import AgentState
from tools.rag_tool import RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Response Writer for an enterprise AI assistant.

You synthesise clear, accurate, and well-structured answers from retrieved
enterprise documents (and optionally images/charts) along with conversation context.

Guidelines:
- Be factual and grounded in the provided context; do NOT hallucinate.
- If charts or images were retrieved, reference them explicitly in your answer.
- Use markdown formatting: headers, bullet points, bold for key terms.
- Cite sources at the end in a "Sources" section (document name / URL).
- If the context is incomplete, honestly state what is known and what is missing.
- Keep the tone professional and concise.
"""


def build_response_writer_llm() -> AzureChatOpenAI:
    """Return a vision-capable model for the response writer."""
    cfg = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=cfg.azure_openai_endpoint,
        api_key=cfg.azure_openai_api_key,
        api_version=cfg.azure_openai_api_version,
        azure_deployment=cfg.azure_vision_deployment,  # gpt-4.1 / gpt-4o
        temperature=0.3,
        max_tokens=2048,
    )


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No retrieved context available."
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {c.source} | Relevance {c.score:.2f}]\n{c.text[:1200]}"
        )
    return "\n\n---\n\n".join(parts)


def _build_image_content_blocks(fetched_images: list[dict]) -> list[dict]:
    """
    Build OpenAI-compatible image_url content blocks from base64 images.
    Each dict: {"blob_path": str, "base64": str}
    """
    blocks = []
    for img in fetched_images:
        b64 = img.get("base64", "")
        if not b64:
            continue
        blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    # PNG assumed; adjust media_type if needed
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            }
        )
    return blocks


def response_writer_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Response Writer Agent.

    Reads  : state.retrieved_chunks, state.fetched_images, state.user_query,
             state.conversation_history_text
    Writes : state.final_response, state.next_node
    """
    logger.info("[ResponseWriter] Building final response")

    llm = build_response_writer_llm()
    cfg = get_settings()

    chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])
    fetched_images: list[dict] = state.get("fetched_images", [])
    history_text = state.get("conversation_history_text", "No prior history.")

    context_block = _build_context_block(chunks)

    # ── Build multimodal user message ──────────────────────────────────────────
    text_content = (
        f"User question: {state['user_query']}\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Retrieved enterprise context:\n{context_block}\n\n"
        "Please synthesise a comprehensive answer using the context above."
    )

    # Start with text block
    message_content: list[dict] = [{"type": "text", "text": text_content}]

    # Append image blocks if any were fetched
    image_blocks = _build_image_content_blocks(fetched_images)
    if image_blocks:
        logger.info("[ResponseWriter] Including %d image(s) in prompt", len(image_blocks))
        message_content.extend(image_blocks)
        # Append image instruction to text
        message_content[0]["text"] += (
            f"\n\nNote: {len(image_blocks)} image(s)/chart(s) have also been "
            "retrieved and are included above. Reference them explicitly in your answer."
        )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=message_content),  # type: ignore[arg-type]
    ]

    response = llm.invoke(messages)
    final_answer = response.content.strip()

    logger.info("[ResponseWriter] Response length: %d chars", len(final_answer))

    return {
        **state,
        "final_response": final_answer,
        "next_node": "end",
        "agent_scratchpad": state.get("agent_scratchpad", [])
        + ["ResponseWriter: answer synthesised"],
    }
