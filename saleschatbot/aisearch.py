"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           AZURE AI SEARCH — COMPLETE RAG IMPLEMENTATION GUIDE              ║
║   Covers: Indexing · Querying · Hybrid Search · Automation · Web Fallback  ║
╚══════════════════════════════════════════════════════════════════════════════╝

PREREQUISITES:
    pip install azure-search-documents azure-identity openai requests python-dotenv

ENVIRONMENT VARIABLES (.env):
    AZURE_SEARCH_ENDPOINT=https://<your-service>.search.windows.net
    AZURE_SEARCH_ADMIN_KEY=<your-admin-key>
    AZURE_SEARCH_QUERY_KEY=<your-query-key>
    AZURE_OPENAI_ENDPOINT=https://<your-openai>.openai.azure.com/
    AZURE_OPENAI_KEY=<your-openai-key>
    AZURE_OPENAI_DEPLOYMENT=gpt-4o          # chat model deployment name
    AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-large
    BING_SEARCH_KEY=<your-bing-key>         # for web fallback (optional)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
import os, json, time, hashlib, datetime, asyncio
from typing import Optional
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType,
    SimpleField, SearchableField, ComplexField,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SemanticConfiguration, SemanticSearch, SemanticPrioritizedFields,
    SemanticField, SearchIndexerDataSourceConnection,
    SearchIndexerDataSourceType, SearchIndexerDataContainer,
    SearchIndexer, IndexingSchedule, FieldMapping,
    SearchIndexerSkillset, OcrSkill, MergeSkill, SplitSkill,
    EntityRecognitionSkill, KeyPhraseExtractionSkill,
    InputFieldMappingEntry, OutputFieldMappingEntry,
    WebApiSkill, SearchIndexerKnowledgeStore,
)
from azure.search.documents.models import VectorizedQuery
import openai
import requests

load_dotenv()

# ── Connection constants ──────────────────────────────────────────────────────
SEARCH_ENDPOINT   = os.getenv("AZURE_SEARCH_ENDPOINT", "https://YOUR-SERVICE.search.windows.net")
SEARCH_ADMIN_KEY  = os.getenv("AZURE_SEARCH_ADMIN_KEY", "YOUR-ADMIN-KEY")
SEARCH_QUERY_KEY  = os.getenv("AZURE_SEARCH_QUERY_KEY", "YOUR-QUERY-KEY")
OPENAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT",  "https://YOUR-OAI.openai.azure.com/")
OPENAI_KEY        = os.getenv("AZURE_OPENAI_KEY",        "YOUR-OAI-KEY")
CHAT_DEPLOYMENT   = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
EMBED_DEPLOYMENT  = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")
BING_KEY          = os.getenv("BING_SEARCH_KEY", "")
INDEX_NAME        = "rag-knowledge-base"

# ── SDK clients ───────────────────────────────────────────────────────────────
admin_cred  = AzureKeyCredential(SEARCH_ADMIN_KEY)
query_cred  = AzureKeyCredential(SEARCH_QUERY_KEY)

index_client    = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=admin_cred)
indexer_client  = SearchIndexerClient(endpoint=SEARCH_ENDPOINT, credential=admin_cred)

openai_client   = openai.AzureOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version="2024-05-01-preview",
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CREATE THE INDEX SCHEMA
#    Includes: keyword fields · dense vector field · semantic config
# ─────────────────────────────────────────────────────────────────────────────
def create_index() -> None:
    """
    FEATURE: Index Schema Design
    ─────────────────────────────
    An Azure AI Search index is the container for your RAG knowledge base.
    Key field types:
      • SimpleField       – filterable/sortable, not full-text searchable
      • SearchableField   – full-text searchable (BM25 keyword ranking)
      • SearchField(Collection(Single)) – stores the dense embedding vector
    """

    fields = [
        # ── Identity & metadata ────────────────────────────────────────────
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,                          # must be unique per document
            filterable=True,
        ),
        SimpleField(
            name="source_url",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=False,
        ),
        SimpleField(
            name="ingested_at",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="category",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,                    # enables category drill-down
        ),

        # ── Content fields (full-text searchable) ─────────────────────────
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",      # language-aware tokeniser
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),
        SearchableField(
            name="key_phrases",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
        ),

        # ── Dense vector field (for semantic/hybrid search) ────────────────
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,     # text-embedding-3-large output dim
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    # ── Vector search config (HNSW approximate nearest-neighbour) ──────────
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-algo",
                parameters={
                    "m": 4,             # graph connectivity (higher = more accurate, slower)
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine", # cosine | dotProduct | euclidean
                },
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-algo",
            )
        ],
    )

    # ── Semantic search config (BM25 + cross-encoder re-ranking) ───────────
    semantic_config = SemanticConfiguration(
        name="default-semantic",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            content_fields=[SemanticField(field_name="content")],
            keywords_fields=[SemanticField(field_name="key_phrases")],
        ),
    )
    semantic_search = SemanticSearch(configurations=[semantic_config])

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    result = index_client.create_or_update_index(index)
    print(f"[✓] Index '{result.name}' created/updated.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMBEDDING HELPER
# ─────────────────────────────────────────────────────────────────────────────
def embed(text: str) -> list[float]:
    """
    FEATURE: Dense Embeddings via Azure OpenAI
    ───────────────────────────────────────────
    text-embedding-3-large → 3072-dim float32 vector.
    These vectors capture semantic meaning so that "car" and "automobile"
    are close together in vector space, enabling semantic search.
    """
    response = openai_client.embeddings.create(
        model=EMBED_DEPLOYMENT,
        input=text,
    )
    return response.data[0].embedding


# ─────────────────────────────────────────────────────────────────────────────
# 3. MANUAL / PROGRAMMATIC INDEXING
# ─────────────────────────────────────────────────────────────────────────────
def index_documents(docs: list[dict]) -> None:
    """
    FEATURE: Push-based indexing
    ─────────────────────────────
    Suitable for: databases, APIs, custom data pipelines.
    Each document must match the index schema.
    The SDK batches up to 1,000 docs per request automatically.
    """
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=query_cred,
    )

    enriched = []
    for doc in docs:
        # Build a deterministic ID from the URL so re-ingestion is idempotent
        doc_id = hashlib.md5(doc["source_url"].encode()).hexdigest()
        vector  = embed(doc["content"])

        enriched.append({
            "id":             doc_id,
            "source_url":     doc["source_url"],
            "ingested_at":    datetime.datetime.utcnow().isoformat() + "Z",
            "category":       doc.get("category", "general"),
            "title":          doc.get("title", ""),
            "content":        doc["content"],
            "key_phrases":    doc.get("key_phrases", []),
            "content_vector": vector,
        })

    # merge_or_upload: updates existing docs, inserts new ones
    result = search_client.merge_or_upload_documents(documents=enriched)
    succeeded = sum(1 for r in result if r.succeeded)
    print(f"[✓] Indexed {succeeded}/{len(enriched)} documents.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. AUTOMATED INDEXING VIA INDEXER + DATA SOURCE
#    Supports: Azure Blob Storage, SQL Database, Cosmos DB, SharePoint …
# ─────────────────────────────────────────────────────────────────────────────
def setup_blob_indexer(
    blob_connection_string: str,
    container_name: str,
) -> None:
    """
    FEATURE: Automated Pull-based Indexing (Indexer)
    ──────────────────────────────────────────────────
    Azure AI Search Indexers crawl your data source on a schedule.
    New/modified documents are automatically picked up.

    Supported data sources:
      azureblob | azuretable | azuresql | cosmosdb |
      adlsgen2  | sharepoint | mysql    | mongodb

    Schedule options:
      PT5M (every 5 min) | PT1H | P1D | P7D
    """

    # ── Step 1: Register the data source ──────────────────────────────────
    data_source = SearchIndexerDataSourceConnection(
        name="blob-datasource",
        type=SearchIndexerDataSourceType.AZURE_BLOB,
        connection_string=blob_connection_string,
        container=SearchIndexerDataContainer(name=container_name),
    )
    indexer_client.create_or_update_data_source_connection(data_source)
    print("[✓] Data source registered.")

    # ── Step 2: Define an AI-enrichment Skillset ───────────────────────────
    #    Skills transform raw content before it lands in the index.
    skillset = SearchIndexerSkillset(
        name="rag-skillset",
        description="OCR → Merge → Split → KeyPhrases → EntityRecognition",
        skills=[

            # OCR: extract text from scanned PDFs / images
            OcrSkill(
                name="ocr",
                description="Extract text from images",
                context="/document/normalized_images/*",
                inputs=[InputFieldMappingEntry(name="image",  source="/document/normalized_images/*")],
                outputs=[OutputFieldMappingEntry(name="text", target_name="text")],
            ),

            # Merge OCR text with any existing text
            MergeSkill(
                name="merge",
                context="/document",
                inputs=[
                    InputFieldMappingEntry(name="text",            source="/document/content"),
                    InputFieldMappingEntry(name="itemsToInsert",   source="/document/normalized_images/*/text"),
                ],
                outputs=[OutputFieldMappingEntry(name="mergedText", target_name="merged_content")],
            ),

            # Split long docs into overlapping ~512-token chunks
            SplitSkill(
                name="split",
                context="/document",
                text_split_mode="pages",          # pages | sentences
                maximum_page_length=512,
                page_overlap_length=64,           # overlap avoids losing context at chunk edges
                inputs=[InputFieldMappingEntry(name="text", source="/document/merged_content")],
                outputs=[OutputFieldMappingEntry(name="textItems", target_name="pages")],
            ),

            # Extract key phrases for the semantic config keywords_fields
            KeyPhraseExtractionSkill(
                name="keyphrases",
                context="/document/pages/*",
                inputs=[InputFieldMappingEntry(name="text", source="/document/pages/*")],
                outputs=[OutputFieldMappingEntry(name="keyPhrases", target_name="key_phrases")],
            ),

            # Named entity recognition (people, orgs, locations…)
            EntityRecognitionSkill(
                name="ner",
                context="/document/pages/*",
                categories=["Person", "Organization", "Location"],
                inputs=[InputFieldMappingEntry(name="text", source="/document/pages/*")],
                outputs=[OutputFieldMappingEntry(name="entities", target_name="entities")],
            ),
        ],
    )
    indexer_client.create_or_update_skillset(skillset)
    print("[✓] Skillset registered.")

    # ── Step 3: Create the Indexer (wires datasource → skillset → index) ──
    indexer = SearchIndexer(
        name="blob-indexer",
        data_source_name="blob-datasource",
        skillset_name="rag-skillset",
        target_index_name=INDEX_NAME,
        schedule=IndexingSchedule(interval=datetime.timedelta(hours=1)),   # run every hour
        field_mappings=[
            FieldMapping(source_field_name="metadata_storage_path", target_field_name="source_url"),
            FieldMapping(source_field_name="metadata_storage_name", target_field_name="title"),
        ],
        output_field_mappings=[
            FieldMapping(source_field_name="/document/pages/*/key_phrases", target_field_name="key_phrases"),
        ],
        parameters={
            "configuration": {
                "dataToExtract": "contentAndMetadata",
                "imageAction": "generateNormalizedImages",   # enables OCR
                "parsingMode": "default",                    # default | json | jsonArray | text
            }
        },
    )
    indexer_client.create_or_update_indexer(indexer)
    print("[✓] Indexer created. First run will start shortly.")


def run_indexer_now(indexer_name: str = "blob-indexer") -> None:
    """Trigger an on-demand indexer run (outside the schedule)."""
    indexer_client.run_indexer(indexer_name)
    print(f"[✓] Indexer '{indexer_name}' triggered.")


def get_indexer_status(indexer_name: str = "blob-indexer") -> dict:
    """Poll indexer status — useful in CI/CD pipelines."""
    status = indexer_client.get_indexer_status(indexer_name)
    last = status.last_result
    return {
        "status": status.status.value,
        "last_run_status": last.status.value if last else "never",
        "docs_succeeded": last.item_count if last else 0,
        "docs_failed":    last.failed_item_count if last else 0,
        "errors":         [e.error_message for e in (last.errors or [])],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. SEARCH MODES
# ─────────────────────────────────────────────────────────────────────────────

def keyword_search(query: str, top: int = 5) -> list[dict]:
    """
    FEATURE: Keyword / BM25 Search
    ────────────────────────────────
    Classic inverted-index search. Great for exact-match queries.
    Uses OKapi BM25 scoring (TF-IDF variant).
    """
    client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, query_cred)
    results = client.search(
        search_text=query,
        top=top,
        include_total_count=True,
        highlight_fields="content",           # wraps matched terms in <em>
        highlight_pre_tag="**",
        highlight_post_tag="**",
    )
    return [{"title": r["title"], "content": r["content"], "score": r["@search.score"]}
            for r in results]


def vector_search(query: str, top: int = 5) -> list[dict]:
    """
    FEATURE: Pure Vector (Semantic Similarity) Search
    ───────────────────────────────────────────────────
    Embeds the query, then finds the nearest neighbours by cosine similarity.
    Returns conceptually similar documents even with zero keyword overlap.
    """
    client  = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, query_cred)
    q_vec   = embed(query)
    results = client.search(
        search_text=None,                     # pure vector — no keyword layer
        vector_queries=[
            VectorizedQuery(
                vector=q_vec,
                k_nearest_neighbors=top,
                fields="content_vector",
            )
        ],
        top=top,
    )
    return [{"title": r["title"], "content": r["content"], "score": r["@search.score"]}
            for r in results]


def hybrid_search(query: str, top: int = 5) -> list[dict]:
    """
    FEATURE: Hybrid Search (BM25 + Vector, fused via RRF)
    ───────────────────────────────────────────────────────
    Reciprocal Rank Fusion (RRF) merges keyword and vector rankings.
    Best of both worlds: exact match + semantic understanding.
    This is the RECOMMENDED mode for production RAG pipelines.
    """
    client  = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, query_cred)
    q_vec   = embed(query)
    results = client.search(
        search_text=query,                    # BM25 leg
        vector_queries=[
            VectorizedQuery(
                vector=q_vec,
                k_nearest_neighbors=top,
                fields="content_vector",
            )
        ],
        top=top,
    )
    return [{"title": r["title"], "content": r["content"], "score": r["@search.score"]}
            for r in results]


def semantic_reranked_hybrid_search(query: str, top: int = 5) -> list[dict]:
    """
    FEATURE: Semantic Reranking on top of Hybrid Search
    ─────────────────────────────────────────────────────
    After hybrid retrieval, Azure's cross-encoder re-ranks results.
    Also generates captions and answers (extractive).
    This is the highest-quality retrieval mode.

    Requires: Semantic Search pricing tier (Standard S1+)
    """
    client  = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, query_cred)
    q_vec   = embed(query)
    results = client.search(
        search_text=query,
        vector_queries=[
            VectorizedQuery(vector=q_vec, k_nearest_neighbors=top, fields="content_vector")
        ],
        query_type="semantic",
        semantic_configuration_name="default-semantic",
        query_caption="extractive",           # pulls the most relevant sentence snippet
        query_answer="extractive",            # tries to answer the question directly
        top=top,
    )

    hits = []
    for r in results:
        caption = r.get("@search.captions", [{}])
        hits.append({
            "title":         r["title"],
            "content":       r["content"],
            "rerank_score":  r.get("@search.reranker_score"),
            "caption":       caption[0].get("text", "") if caption else "",
        })

    # Extractive answer (if Azure found a direct answer)
    answers = results.get_answers()
    if answers:
        print(f"\n[Extractive Answer]: {answers[0].text}\n")

    return hits


def faceted_filtered_search(
    query: str,
    category: str,
    ingested_after: str = "2024-01-01T00:00:00Z",
    top: int = 5,
) -> list[dict]:
    """
    FEATURE: Faceted Navigation & OData Filters
    ─────────────────────────────────────────────
    Filters narrow the result set BEFORE scoring (cheaper than post-filtering).
    Facets let users drill down by category (like an e-commerce sidebar).

    OData filter syntax examples:
      "category eq 'finance'"
      "ingested_at ge 2024-01-01T00:00:00Z"
      "category eq 'legal' and ingested_at ge 2023-01-01T00:00:00Z"
    """
    client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, query_cred)
    results = client.search(
        search_text=query,
        filter=f"category eq '{category}' and ingested_at ge {ingested_after}",
        facets=["category,count:10"],         # return category counts for sidebar
        top=top,
    )
    facets = results.get_facets()
    print(f"[Facets] {facets}")
    return [{"title": r["title"], "content": r["content"]} for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# 6. WEB SEARCH FALLBACK (Bing Search API)
#    Triggered when the AI Search index has no relevant results
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_score(results: list[dict]) -> float:
    """Heuristic: average of top-3 rerank scores (0–4 scale)."""
    scores = [r.get("rerank_score") or r.get("score", 0) for r in results[:3]]
    return sum(scores) / len(scores) if scores else 0.0


def web_search_fallback(query: str, top: int = 3) -> list[dict]:
    """
    FEATURE: Live Web Search Fallback via Bing Search API
    ───────────────────────────────────────────────────────
    When the index doesn't have the answer, fetch from the web and
    optionally index the result for future queries (auto-learning).

    Requires: Bing Search v7 resource (Azure Marketplace)
    """
    if not BING_KEY:
        print("[!] BING_SEARCH_KEY not set — skipping web fallback.")
        return []

    headers = {"Ocp-Apim-Subscription-Key": BING_KEY}
    params  = {"q": query, "count": top, "responseFilter": "Webpages", "mkt": "en-US"}
    resp    = requests.get("https://api.bing.microsoft.com/v7.0/search",
                           headers=headers, params=params, timeout=10)
    resp.raise_for_status()

    pages = resp.json().get("webPages", {}).get("value", [])
    return [
        {
            "title":      p["name"],
            "content":    p["snippet"],
            "source_url": p["url"],
            "category":   "web-fallback",
        }
        for p in pages
    ]


def index_web_results(web_docs: list[dict]) -> None:
    """
    Index web fallback results back into the knowledge base
    so the same question won't fall back to the web next time.
    This creates a SELF-IMPROVING RAG pipeline.
    """
    if web_docs:
        index_documents(web_docs)
        print(f"[✓] {len(web_docs)} web results indexed for future queries.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. THE COMPLETE RAG PIPELINE
#    Retrieve → (web fallback if needed) → Augment → Generate
# ─────────────────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 1.5   # reranker score: tune based on your data

def rag_answer(
    user_question: str,
    top: int = 5,
    auto_index_web_results: bool = True,
    chat_history: Optional[list[dict]] = None,
) -> dict:
    """
    FEATURE: Full RAG Pipeline with Web Fallback
    ──────────────────────────────────────────────
    Pipeline stages:
      1. Hybrid + semantic retrieval from Azure AI Search
      2. Confidence check → web fallback if score too low
      3. Build grounded prompt (context + conversation history)
      4. Azure OpenAI GPT-4o generates the answer
      5. Citations extracted from source_url fields
      6. Optionally index web results for future queries

    Returns:
      {
        "answer":    str,
        "sources":   list[str],
        "from_web":  bool,
        "contexts":  list[dict],
      }
    """
    chat_history = chat_history or []

    # ── Stage 1: Retrieve from index ───────────────────────────────────────
    print(f"\n[RAG] Retrieving for: '{user_question}'")
    results = semantic_reranked_hybrid_search(user_question, top=top)
    confidence = _confidence_score(results)
    print(f"[RAG] Confidence score: {confidence:.2f} (threshold: {CONFIDENCE_THRESHOLD})")

    # ── Stage 2: Web fallback if index confidence is low ──────────────────
    from_web = False
    if confidence < CONFIDENCE_THRESHOLD or not results:
        print("[RAG] Low confidence → triggering web fallback...")
        web_docs = web_search_fallback(user_question, top=3)
        if web_docs:
            from_web = True
            results  = [{"title": d["title"], "content": d["content"],
                         "source_url": d["source_url"]} for d in web_docs]
            if auto_index_web_results:
                index_web_results(web_docs)

    # ── Stage 3: Build the grounded system prompt ─────────────────────────
    context_blocks = []
    for i, r in enumerate(results, 1):
        url = r.get("source_url", "internal index")
        context_blocks.append(f"[Source {i}: {url}]\n{r['content']}")

    context_text = "\n\n---\n\n".join(context_blocks)

    system_prompt = f"""You are a precise, grounded AI assistant.
Answer ONLY based on the provided context. If the context doesn't contain
the answer, say so clearly — do not hallucinate.

Always cite the source number(s) you used, e.g. "According to [Source 2]..."

Context:
{context_text}
"""

    # ── Stage 4: Generate with GPT-4o ────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]
    # Inject conversation history for multi-turn RAG
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_question})

    response = openai_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=messages,
        temperature=0.2,        # low temperature = more factual
        max_tokens=1024,
    )
    answer = response.choices[0].message.content

    # ── Stage 5: Extract citations ────────────────────────────────────────
    sources = list({r.get("source_url", "") for r in results if r.get("source_url")})

    return {
        "answer":   answer,
        "sources":  sources,
        "from_web": from_web,
        "contexts": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. MULTI-TURN CONVERSATIONAL RAG
# ─────────────────────────────────────────────────────────────────────────────

class ConversationalRAG:
    """
    FEATURE: Stateful Multi-Turn RAG
    ──────────────────────────────────
    Maintains chat history so follow-up questions have context.
    Uses query rewriting to make each turn self-contained before retrieval.
    """

    def __init__(self):
        self.history: list[dict] = []

    def _rewrite_query(self, user_input: str) -> str:
        """Rewrite a follow-up question into a standalone query for better retrieval."""
        if not self.history:
            return user_input

        rewrite_prompt = (
            "Given this conversation history:\n"
            + "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.history[-4:])
            + f"\n\nUser's follow-up: {user_input}\n\n"
            "Rewrite the follow-up as a fully self-contained search query "
            "that doesn't need the history to be understood. Output ONLY the rewritten query."
        )
        resp = openai_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0,
            max_tokens=128,
        )
        return resp.choices[0].message.content.strip()

    def chat(self, user_input: str) -> str:
        standalone_query = self._rewrite_query(user_input)
        print(f"[ConvRAG] Standalone query: '{standalone_query}'")

        result = rag_answer(
            user_question=standalone_query,
            chat_history=self.history,
        )
        answer = result["answer"]

        # Append to history for next turn
        self.history.append({"role": "user",      "content": user_input})
        self.history.append({"role": "assistant",  "content": answer})

        return answer


# ─────────────────────────────────────────────────────────────────────────────
# 9. ADVANCED FEATURES SHOWCASE
# ─────────────────────────────────────────────────────────────────────────────

def autocomplete_and_suggest(partial_query: str) -> dict:
    """
    FEATURE: Autocomplete & Suggestions
    ─────────────────────────────────────
    Autocomplete: completes a partial search term mid-typing.
    Suggest: returns full document titles matching the partial query.
    Requires the Suggester to be defined in the index schema.
    """
    client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, query_cred)

    # Note: requires a suggester named "sg" in the index schema
    try:
        completions = client.autocomplete(
            search_text=partial_query,
            suggester_name="sg",
            mode="twoTerms",
        )
        suggestions = client.suggest(
            search_text=partial_query,
            suggester_name="sg",
            select=["title"],
        )
        return {
            "completions": [c["query_plus_text"] for c in completions],
            "suggestions": [s["title"] for s in suggestions],
        }
    except Exception as e:
        return {"error": str(e), "note": "Suggester must be defined in the index schema."}


def delete_document(doc_id: str) -> None:
    """
    FEATURE: Document Deletion
    ───────────────────────────
    Hard-delete a document from the index by its key.
    """
    client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, query_cred)
    client.delete_documents(documents=[{"id": doc_id}])
    print(f"[✓] Document '{doc_id}' deleted.")


def get_index_stats() -> dict:
    """
    FEATURE: Index Statistics
    ──────────────────────────
    Returns document count and storage size for monitoring.
    """
    stats = index_client.get_index_statistics(INDEX_NAME)
    return {
        "document_count":    stats.document_count,
        "storage_size_bytes": stats.storage_size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10. DEMO — RUNS WHEN EXECUTED DIRECTLY
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  AZURE AI SEARCH RAG — FULL FEATURE DEMO")
    print("=" * 70)

    # ── 1. Create the index ────────────────────────────────────────────────
    print("\n[STEP 1] Creating index schema...")
    create_index()

    # ── 2. Seed some sample documents ─────────────────────────────────────
    print("\n[STEP 2] Indexing sample documents...")
    sample_docs = [
        {
            "source_url": "https://docs.microsoft.com/azure/search/rag-overview",
            "title": "RAG with Azure AI Search Overview",
            "content": (
                "Retrieval-Augmented Generation (RAG) combines information retrieval "
                "with large language model generation. Azure AI Search acts as the "
                "retrieval layer, providing relevant document chunks that ground the "
                "LLM's response and prevent hallucinations."
            ),
            "category": "documentation",
            "key_phrases": ["RAG", "Azure AI Search", "LLM", "grounding"],
        },
        {
            "source_url": "https://docs.microsoft.com/azure/search/hybrid-search",
            "title": "Hybrid Search in Azure AI Search",
            "content": (
                "Hybrid search fuses BM25 keyword scoring and dense vector similarity "
                "using Reciprocal Rank Fusion (RRF). This consistently outperforms "
                "either approach alone, especially for domain-specific vocabularies "
                "where semantic models may not cover all terms."
            ),
            "category": "documentation",
            "key_phrases": ["hybrid search", "BM25", "RRF", "vector search"],
        },
        {
            "source_url": "https://internal/policy/data-retention",
            "title": "Company Data Retention Policy 2025",
            "content": (
                "All customer data must be retained for a minimum of 7 years per "
                "regulatory requirements. Data classified as PII must be encrypted "
                "at rest using AES-256 and access-logged. Deletion requests must be "
                "processed within 30 days under GDPR Article 17."
            ),
            "category": "legal",
            "key_phrases": ["data retention", "PII", "GDPR", "AES-256"],
        },
    ]
    index_documents(sample_docs)

    # ── 3. Keyword search ──────────────────────────────────────────────────
    print("\n[STEP 3] Keyword search: 'GDPR data deletion'")
    kw_results = keyword_search("GDPR data deletion", top=3)
    for r in kw_results:
        print(f"  • [{r['score']:.2f}] {r['title']}")

    # ── 4. Vector search ───────────────────────────────────────────────────
    print("\n[STEP 4] Vector search: 'how do I erase personal information'")
    vec_results = vector_search("how do I erase personal information", top=3)
    for r in vec_results:
        print(f"  • [{r['score']:.4f}] {r['title']}")

    # ── 5. Full RAG answer ─────────────────────────────────────────────────
    print("\n[STEP 5] Full RAG answer...")
    question = "What is hybrid search and why should I use it for RAG?"
    result = rag_answer(question)
    print(f"\nQ: {question}")
    print(f"\nA: {result['answer']}")
    print(f"\nSources: {result['sources']}")
    print(f"From web: {result['from_web']}")

    # ── 6. Conversational RAG ──────────────────────────────────────────────
    print("\n[STEP 6] Conversational RAG demo...")
    conv = ConversationalRAG()
    a1 = conv.chat("What is RAG?")
    print(f"Turn 1: {a1[:200]}...")
    a2 = conv.chat("How does that relate to Azure AI Search specifically?")
    print(f"Turn 2: {a2[:200]}...")

    # ── 7. Index stats ─────────────────────────────────────────────────────
    print("\n[STEP 7] Index statistics:")
    stats = get_index_stats()
    print(f"  Documents: {stats['document_count']}")
    print(f"  Storage:   {stats['storage_size_bytes']:,} bytes")

    print("\n" + "=" * 70)
    print("  Demo complete. See comments in each function for full feature docs.")
    print("=" * 70)