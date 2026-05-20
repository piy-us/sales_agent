from langchain.agents import create_agent

from langchain_openai import (
    AzureChatOpenAI
)

from app.core.config import settings


SYSTEM_PROMPT = """
You are a helpful AI sales assistant.
"""


def build_sales_agent():

    llm = AzureChatOpenAI(

        api_key=settings.AZURE_OPENAI_KEY,

        azure_endpoint=(
            settings.AZURE_OPENAI_ENDPOINT
        ),

        azure_deployment=(
            settings
            .AZURE_OPENAI_CHAT_DEPLOYMENT
        ),

        api_version="2024-02-15-preview",

        temperature=0.3,
    )

    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )

    return agent