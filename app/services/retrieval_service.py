from azure.core.credentials import (
    AzureKeyCredential
)

from azure.search.documents.aio import (
    SearchClient
)

from azure.search.documents.models import (
    VectorizedQuery
)

from langchain_openai import (
    AzureOpenAIEmbeddings
)

from app.core.config import settings


class RetrievalService:

    def __init__(self):

        self.embeddings = (
            AzureOpenAIEmbeddings(

                api_key=(
                    settings.AZURE_OPENAI_KEY
                ),

                azure_endpoint=(
                    settings
                    .AZURE_OPENAI_ENDPOINT
                ),

                azure_deployment=(
                    settings
                    .AZURE_OPENAI_EMBEDDING_DEPLOYMENT
                ),

                api_version="2024-02-15-preview",
            )
        )

        self.search_client = SearchClient(

            endpoint=(
                settings.AZURE_SEARCH_ENDPOINT
            ),

            index_name=(
                settings.AZURE_SEARCH_INDEX
            ),

            credential=AzureKeyCredential(
                settings.AZURE_SEARCH_KEY
            ),
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ):

        embedding = await (
            self.embeddings.aembed_query(
                query
            )
        )

        vector_query = VectorizedQuery(
            vector=embedding,
            k_nearest_neighbors=top_k,
            fields="embedding",
        )

        results = await self.search_client.search(

            search_text=query,

            vector_queries=[vector_query],

            top=top_k,
        )

        documents = []

        async for result in results:

            documents.append({
                "content": result["content"],
                "source": result.get(
                    "source"
                ),
                "score": result.get(
                    "@search.score"
                ),
            })

        return documents