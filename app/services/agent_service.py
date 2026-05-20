from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from app.agents.sales_agent import (
    build_sales_agent
)


class AgentService:

    def __init__(self):

        self.agent = build_sales_agent()

    async def generate_response(
        self,
        user_query: str,
        conversation_history: list,
        rag_context: str,
    ):

        messages = []

        # HISTORY
        for msg in conversation_history:

            if msg["role"] == "user":

                messages.append(
                    HumanMessage(
                        content=msg["content"]
                    )
                )

            elif msg["role"] == "assistant":

                messages.append(
                    AIMessage(
                        content=msg["content"]
                    )
                )

        # RAG CONTEXT
        if rag_context:

            messages.append(
                SystemMessage(
                    content=f"""
Retrieved Knowledge:

{rag_context}
"""
                )
            )

        # CURRENT QUERY
        messages.append(
            HumanMessage(
                content=user_query
            )
        )

        result = await self.agent.ainvoke({
            "messages": messages
        })

        final_message = result["messages"][-1]

        return str(final_message.content)