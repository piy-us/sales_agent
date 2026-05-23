# Multi-Agent LangGraph Orchestration Chatbot

## Architecture

```
User Query
    │
    ▼
[Conversation Agent]  ──── Greeting/Explanation ──── Direct Response
    │
    │ (complex query)
    ▼
[Query Rewriter Agent]  ← CosmosDB History
    │
    ▼
[RAG / Azure AI Search (placeholder)]
    │
    ▼
[Context Planner Agent]
    │ ├── enough context ──► [Response Writer Agent] ──► User
    │ └── insufficient   ──► [Query Rewriter Agent] (loop)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your Azure credentials in .env
python api/server.py
```

## Environment Variables

See `.env.example` for all required variables.

## Project Structure

```
├── agents/
│   ├── conversation_agent.py     # Routes queries or responds directly
│   ├── query_rewriter_agent.py   # Rewrites query using history for RAG
│   ├── context_planner_agent.py  # Evaluates retrieved context sufficiency
│   └── response_writer_agent.py  # Synthesises final response (vision LLM)
├── graph/
│   └── orchestrator.py           # LangGraph StateGraph definition
├── tools/
│   ├── rag_tool.py               # Placeholder → Azure AI Search
│   └── blob_tool.py              # Placeholder → Azure Blob Storage images
├── memory/
│   └── cosmos_memory.py          # Azure CosmosDB conversation history
├── config/
│   └── settings.py               # Centralised config from env
├── api/
│   └── server.py                 # FastAPI server
└── requirements.txt
```
