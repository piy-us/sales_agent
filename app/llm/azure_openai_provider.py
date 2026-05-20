from openai import AsyncAzureOpenAI

from app.core.config import settings

from app.llm.base import BaseLLMProvider

from app.llm.models import (
    LLMMessage,
    GenerationConfig
)


class AzureOpenAIProvider(
    BaseLLMProvider
):

    def __init__(self):

        self.client = AsyncAzureOpenAI(

            api_key=settings.AZURE_OPENAI_KEY,

            azure_endpoint=(
                settings.AZURE_OPENAI_ENDPOINT
            ),

            api_version="2024-02-15-preview",
        )

    async def generate(
        self,
        messages: list[LLMMessage],
        config: GenerationConfig
    ) -> str:

        response = (
            await self.client.chat.completions.create(

                model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,

                messages=[
                    {
                        "role": m.role,
                        "content": m.content
                    }
                    for m in messages
                ],

                temperature=config.temperature,

                max_tokens=config.max_tokens,
            )
        )

        return (
            response.choices[0]
            .message
            .content
        )

    async def embed(
        self,
        text: str
    ) -> list[float]:

        response = (
            await self.client.embeddings.create(

                model=(
                    settings
                    .AZURE_OPENAI_EMBEDDING_DEPLOYMENT
                ),

                input=text,
            )
        )

        return response.data[0].embedding