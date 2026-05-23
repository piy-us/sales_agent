"""
api/server.py
──────────────
FastAPI application exposing the multi-agent chatbot over HTTP.

Endpoints
─────────
POST /chat                 – send a message, get a response
GET  /history/{session_id} – fetch conversation history
DELETE /history/{session_id} – clear conversation history
GET  /health               – liveness probe

Run locally:
    uvicorn api.server:app --reload --port 8000

Or directly:
    python api/server.py
"""
from __future__ import annotations

import logging
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Bootstrap logging ──────────────────────────────────────────────────────────
from config.settings import get_settings

cfg = get_settings()
logging.basicConfig(
    level=getattr(logging, cfg.log_level.upper(), logging.INFO),
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lazy imports (heavy; only initialise after startup) ────────────────────────
from memory.cosmos_memory import CosmosConversationMemory
from graph.orchestrator import get_graph
from graph.state import AgentState


# ── Lifespan: warm up connections ─────────────────────────────────────────────

_memory: CosmosConversationMemory | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _memory
    logger.info("Starting up – initialising CosmosDB memory and LangGraph …")
    _memory = CosmosConversationMemory()
    _ = get_graph()  # compile graph once at startup
    logger.info("Startup complete.")
    yield
    logger.info("Shutdown complete.")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent Enterprise Chatbot",
    description=(
        "LangGraph-orchestrated multi-agent RAG chatbot "
        "backed by Azure AI Foundry models and CosmosDB memory."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's message")
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation session ID. Omit to start a new session.",
    )


class ChatResponse(BaseModel):
    session_id: str
    response: str
    rewritten_query: Optional[str] = None
    sources: list[str] = []
    agent_trace: list[str] = []
    rewrite_count: int = 0


class HistoryResponse(BaseModel):
    session_id: str
    turns: list[dict]


# ── Helper ─────────────────────────────────────────────────────────────────────

def _get_memory() -> CosmosConversationMemory:
    if _memory is None:
        raise RuntimeError("Memory not initialised – lifespan not running?")
    return _memory


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint.

    - Creates a new session_id if none is provided.
    - Loads conversation history from CosmosDB.
    - Runs the LangGraph multi-agent pipeline.
    - Persists the exchange back to CosmosDB.
    - Returns the assistant's response with optional metadata.
    """
    session_id = req.session_id or str(uuid.uuid4())
    logger.info("Chat request | session=%s query=%r", session_id, req.query)

    memory = _get_memory()
    graph = get_graph()

    # Load history from CosmosDB
    history_text = memory.format_history_for_prompt(session_id)

    # Build initial graph state
    initial_state: AgentState = {
        "session_id": session_id,
        "user_query": req.query,
        "conversation_history_text": history_text,
        "rewrite_count": 0,
        "retrieved_chunks": [],
        "fetched_images": [],
        "agent_scratchpad": [],
        "planner_feedback": "",
        "rewritten_query": "",
        "final_response": "",
        "next_node": "",
        "error": "",
    }

    try:
        final_state: AgentState = graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("Graph invocation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    answer = final_state.get("final_response", "")
    if not answer:
        answer = "I'm sorry, I wasn't able to generate a response. Please try again."

    # Persist exchange to CosmosDB
    try:
        memory.add_exchange(
            session_id=session_id,
            user_content=req.query,
            assistant_content=answer,
        )
    except Exception as exc:
        logger.warning("Failed to save exchange to CosmosDB: %s", exc)

    # Extract sources from retrieved chunks
    sources = list(
        {c.source for c in final_state.get("retrieved_chunks", [])}
    )

    return ChatResponse(
        session_id=session_id,
        response=answer,
        rewritten_query=final_state.get("rewritten_query"),
        sources=sources,
        agent_trace=final_state.get("agent_scratchpad", []),
        rewrite_count=final_state.get("rewrite_count", 0),
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Return the conversation history for a session."""
    memory = _get_memory()
    turns = memory.get_history(session_id)
    return HistoryResponse(session_id=session_id, turns=turns)


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Delete all conversation history for a session."""
    memory = _get_memory()
    memory.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=cfg.log_level.lower(),
    )
