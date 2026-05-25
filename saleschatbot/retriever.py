"""
retriever.py
─────────────────────────────────────────────────────────────────────────────
STEP 6  Query-time: hybrid retrieval (vector + BM25) from Azure AI Search
        followed by a RAG answer using AzureChatOpenAI (GPT-4o) via the
        Azure AI Foundry endpoint  services.ai.azure.com/v1.

        The vision tool lets the LLM fetch & analyse a slide image on-demand
        when the text description alone is not sufficient.

pip install azure-search-documents openai
"""

from __future__ import annotations

import os
import urllib.request
from typing import Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

from uploader import get_embedding   # reuse the same embedding fn

# ── Config ─────────────────────────────────────────────────────────────────
AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_KEY      = os.environ["AZURE_SEARCH_KEY"]

AZURE_FOUNDRY_ENDPOINT = os.environ.get(
    "AZURE_FOUNDRY_ENDPOINT", "https://services.ai.azure.com/v1"
)
AZURE_OPENAI_KEY     = os.environ["AZURE_OPENAI_KEY"]
AZURE_OPENAI_API_VER = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
CHAT_DEPLOYMENT      = os.environ.get("AZURE_CHAT_DEPLOYMENT", "gpt-4o")

INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "pptx-rag-index")

# ── Lazy singleton ─────────────────────────────────────────────────────────
_chat_client: Optional[AzureOpenAI] = None


def _get_chat_client() -> AzureOpenAI:
    global _chat_client
    if _chat_client is None:
        _chat_client = AzureOpenAI(
            azure_endpoint = AZURE_FOUNDRY_ENDPOINT,
            api_key        = AZURE_OPENAI_KEY,
            api_version    = AZURE_OPENAI_API_VER,
        )
    return _chat_client


# ═══════════════════════════════════════════════════════════════════
# Retrieval — hybrid vector + BM25
# ═══════════════════════════════════════════════════════════════════

def retrieve(
    query:       str,
    top_k:       int            = 5,
    filter_expr: str | None     = None,
) -> list[dict]:
    """
    Hybrid search: dense vector (HNSW) + BM25 keyword search via Azure AI
    Search Reciprocal Rank Fusion.

    Parameters
    ----------
    query        : natural-language question
    top_k        : number of results to return
    filter_expr  : optional OData filter, e.g. "source_file eq 'deck.pptx'"

    Returns
    -------
    List of result dicts with keys: id, text, content_type, slide_title,
    slide_number, slide_number_end, source_file, image_uri.
    """
    search_client = SearchClient(
        endpoint   = AZURE_SEARCH_ENDPOINT,
        index_name = INDEX_NAME,
        credential = AzureKeyCredential(AZURE_SEARCH_KEY),
    )

    vector_query = VectorizedQuery(
        vector              = get_embedding(query),
        k_nearest_neighbors = top_k,
        fields              = "embedding",
    )

    results = search_client.search(
        search_text    = query,            # BM25 keyword leg
        vector_queries = [vector_query],   # vector leg
        filter         = filter_expr,
        top            = top_k,
        select         = [
            "id", "text", "content_type",
            "slide_title", "slide_number", "slide_number_end",
            "source_file", "image_uri",
        ],
    )
    return [dict(r) for r in results]


# ═══════════════════════════════════════════════════════════════════
# Vision tool definition (OpenAI function-calling format)
# ═══════════════════════════════════════════════════════════════════

_VISION_TOOL: dict = {
    "type": "function",
    "function": {
        "name":        "fetch_slide_image",
        "description": (
            "Fetch the actual image from a slide when the text description is "
            "not sufficient to answer the user's question precisely. Use this "
            "when the user asks about specific values, colours, or fine details "
            "in a chart or diagram."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_uri": {
                    "type":        "string",
                    "description": "The image URI from the retrieved chunk metadata.",
                },
                "question": {
                    "type":        "string",
                    "description": "The specific question to answer by analysing this image.",
                },
            },
            "required": ["image_uri", "question"],
        },
    },
}


def _fetch_and_analyse_image(image_uri: str, question: str) -> str:
    """
    Download the blob and ask the vision model to answer the question.
    Imported lazily to avoid a circular dependency on parser.describe_visual.
    """
    from parser import describe_visual   # noqa: PLC0415

    with urllib.request.urlopen(image_uri) as resp:
        blob = resp.read()
    return describe_visual(blob, question, is_chart=True)


# ═══════════════════════════════════════════════════════════════════
# RAG answer pipeline
# ═══════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "You are an enterprise assistant. Answer questions using the provided "
    "slide content. Cite slide numbers when possible. "
    "If a retrieved chunk mentions an image_uri and you need more detail from "
    "that image to answer precisely, call the fetch_slide_image function."
)


def answer_query(
    user_question: str,
    top_k:         int        = 5,
    filter_expr:   str | None = None,
    max_tokens:    int        = 1200,
) -> str:
    """
    Full RAG pipeline:
      1. Retrieve top-k chunks (hybrid search).
      2. Build context string.
      3. Call GPT-4o via Foundry for an answer.
      4. If the model invokes fetch_slide_image, execute it and loop.

    Returns the final answer string.
    """
    chunks = retrieve(user_question, top_k=top_k, filter_expr=filter_expr)

    # ── Build context ─────────────────────────────────────────────
    context_parts: list[str] = []
    for c in chunks:
        slide_range = (
            f"Slides {c['slide_number']}–{c['slide_number_end']}"
            if c.get("slide_number_end") and c["slide_number_end"] != c["slide_number"]
            else f"Slide {c['slide_number']}"
        )
        part = f"[{slide_range} — {c['slide_title']}]\n{c['text']}"
        if c.get("image_uri"):
            part += f"\n[image_uri: {c['image_uri']}]"
        context_parts.append(part)

    context = "\n\n---\n\n".join(context_parts)

    # ── Initial LLM call ─────────────────────────────────────────
    client   = _get_chat_client()
    messages = [
        {"role": "system",  "content": _SYSTEM_PROMPT},
        {
            "role":    "user",
            "content": f"Context:\n{context}\n\nQuestion: {user_question}",
        },
    ]

    response = client.chat.completions.create(
        model      = CHAT_DEPLOYMENT,
        max_tokens = max_tokens,
        tools      = [_VISION_TOOL],
        messages   = messages,
    )

    # ── Agentic tool-use loop ─────────────────────────────────────
    import json

    while response.choices[0].finish_reason == "tool_calls":
        assistant_msg = response.choices[0].message
        messages.append(assistant_msg)   # keep history

        for tool_call in assistant_msg.tool_calls or []:
            if tool_call.function.name != "fetch_slide_image":
                continue

            args        = json.loads(tool_call.function.arguments)
            tool_result = _fetch_and_analyse_image(
                args["image_uri"], args["question"]
            )
            messages.append(
                {
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      tool_result,
                }
            )

        # Follow-up call with tool results appended
        response = client.chat.completions.create(
            model      = CHAT_DEPLOYMENT,
            max_tokens = max_tokens,
            tools      = [_VISION_TOOL],
            messages   = messages,
        )

    return response.choices[0].message.content or ""


# ═══════════════════════════════════════════════════════════════════
# CLI convenience
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
               "What does the revenue chart show for Q3?"

    print(f"\n❓ Question: {question}\n")
    answer = answer_query(question)
    print(f"💬 Answer:\n{answer}")
