from abc import ABC, abstractmethod

from app.llm.models import (
    LLMMessage,
    GenerationConfig
)


class BaseLLMProvider(ABC):

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        config: GenerationConfig
    ) -> str:
        pass

    @abstractmethod
    async def embed(
        self,
        text: str
    ) -> list[float]:
        pass