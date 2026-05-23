"""
test_pipeline.py
─────────────────
Quick smoke test to verify the pipeline runs end-to-end
WITHOUT a running server (calls the graph directly).

Run: python test_pipeline.py

Requires:
  - .env file with valid Azure OpenAI + CosmosDB credentials
  - pip install -r requirements.txt
"""
import sys
import logging
import uuid

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from config.settings import get_settings
from memory.cosmos_memory import CosmosConversationMemory
from graph.orchestrator import build_graph
from graph.state import AgentState


def run_turn(graph, memory: CosmosConversationMemory, session_id: str, query: str) -> str:
    history_text = memory.format_history_for_prompt(session_id)

    state: AgentState = {
        "session_id": session_id,
        "user_query": query,
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

    final = graph.invoke(state)
    answer = final.get("final_response", "(no response)")

    memory.add_exchange(session_id, query, answer)

    print("\n" + "=" * 60)
    print(f"Query   : {query}")
    print(f"Decision: next_node={final.get('next_node')}")
    if final.get("rewritten_query"):
        print(f"Rewrite : {final['rewritten_query']}")
    print(f"Rewrites: {final.get('rewrite_count', 0)}")
    print(f"Answer  : {answer[:500]}")
    print("Trace   :")
    for step in final.get("agent_scratchpad", []):
        print(f"  • {step}")
    return answer


def main():
    cfg = get_settings()
    session_id = str(uuid.uuid4())
    print(f"\nSession ID: {session_id}")
    print(f"Using deployment: {cfg.azure_chat_deployment}")

    graph = build_graph()
    memory = CosmosConversationMemory()

    # Turn 1: greeting → should be handled directly
    run_turn(graph, memory, session_id, "Hello! What can you help me with?")

    # Turn 2: enterprise query → should go through RAG pipeline
    run_turn(graph, memory, session_id, "What are the Q3 revenue figures for the APAC region?")

    # Turn 3: follow-up using pronoun → query rewriter should resolve context
    run_turn(graph, memory, session_id, "Can you compare that to the previous quarter?")

    print("\n✅ Smoke test complete")


if __name__ == "__main__":
    main()
