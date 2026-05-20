from pydantic import BaseModel
from typing import Optional


class LLMMessage(BaseModel):

    role: str
    content: str


class GenerationConfig(BaseModel):

    temperature: float = 0.3

    max_tokens: Optional[int] = None