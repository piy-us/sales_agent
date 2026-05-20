from app.llm.azure_openai_provider import (
    AzureOpenAIProvider
)


def get_llm_provider():

    return AzureOpenAIProvider()