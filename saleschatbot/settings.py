"""
config/settings.py
──────────────────
Single source of truth for all environment-driven configuration.
Uses pydantic-settings so every field is type-checked and documented.
"""
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Azure OpenAI / AI Foundry ──────────────────────────────────────────────
    azure_openai_endpoint: str = Field(..., description="Azure OpenAI resource endpoint")
    azure_openai_api_key: str = Field(..., description="Azure OpenAI API key")
    azure_openai_api_version: str = Field(
        default="2025-01-01-preview", description="API version"
    )

    # Deployment names (set in Azure AI Foundry)
    azure_chat_deployment: str = Field(
        default="gpt-4.1-mini",
        description="Deployment for conversation/planning agents",
    )
    azure_vision_deployment: str = Field(
        default="gpt-4.1",
        description="Vision-capable deployment for response writer",
    )

    # ── CosmosDB ───────────────────────────────────────────────────────────────
    cosmos_endpoint: str = Field(..., description="CosmosDB account URI")
    cosmos_key: str = Field(..., description="CosmosDB primary key")
    cosmos_database: str = Field(default="chatbot_db")
    cosmos_container: str = Field(default="conversations")
    cosmos_ttl: int = Field(default=2_592_000, description="Item TTL in seconds")

    # ── Azure AI Search (placeholder) ─────────────────────────────────────────
    azure_search_endpoint: str = Field(default="", description="AI Search endpoint")
    azure_search_key: str = Field(default="", description="AI Search key")
    azure_search_index: str = Field(default="enterprise-docs")

    # ── Azure Blob Storage (placeholder) ──────────────────────────────────────
    azure_blob_connection_string: str = Field(default="", description="Blob connection string")
    azure_blob_container: str = Field(default="enterprise-images")

    # ── Orchestration tuning ───────────────────────────────────────────────────
    max_history_turns: int = Field(default=10)
    max_rewrite_loops: int = Field(default=3)
    relevance_threshold: float = Field(default=0.70)
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()  # type: ignore[call-arg]
