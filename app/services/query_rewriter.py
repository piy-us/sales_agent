from langchain_openai import (
    AzureChatOpenAI
)

from app.core.config import settings


class QueryRewriter:

    def __init__(self):

        self.llm = AzureChatOpenAI(

            api_key=settings.AZURE_OPENAI_KEY,

            azure_endpoint=(
                settings.AZURE_OPENAI_ENDPOINT
            ),

            azure_deployment=(
                settings
                .AZURE_OPENAI_CHAT_DEPLOYMENT
            ),

            api_version="2024-02-15-preview",

            temperature=0,
        )

    async def rewrite(
        self,
        history: list,
        user_query: str,
    ):

        history_text = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in history[-4:]
        ])

        prompt = f"""
Rewrite the conversation into a standalone
retrieval query.

Conversation:
{history_text}

Latest User Query:
{user_query}
"""

        response = await self.llm.ainvoke(
            prompt
        )

        return response.content