from app.llm.factory import (
    get_llm_provider
)

from app.llm.models import (
    LLMMessage,
    GenerationConfig
)


SYSTEM_PROMPT = """
You are a helpful AI assistant.
"""


class GenerationService:

    def __init__(self):

        self.llm = get_llm_provider()

    async def generate(
        self,
        user_query: str,
        context: str,
        conversation_history: list
    ):

        history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversation_history[-6:]
        ])

        prompt = f"""
Conversation:
{history}

Retrieved Context:
{context}

User Question:
{user_query}
"""

        messages = [

            LLMMessage(
                role="system",
                content=SYSTEM_PROMPT
            ),

            LLMMessage(
                role="user",
                content=prompt
            ),
        ]

        return await self.llm.generate(

            messages=messages,

            config=GenerationConfig(
                temperature=0.3
            )
        )