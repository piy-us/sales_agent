from app.llm.factory import (
    get_llm_provider
)


class AzureAISearchService:

    def __init__(self):

        self.llm = get_llm_provider()

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5
    ):

        embedding = await self.llm.embed(
            query
        )

        # search logic