"""
tools/blob_tool.py
───────────────────
PLACEHOLDER – Azure Blob Storage image fetcher.

Contract
────────
fetch_image_as_base64(blob_path: str) → str   (base64-encoded PNG/JPEG)

When Azure Blob Storage is ready:
  1. Set AZURE_BLOB_CONNECTION_STRING and AZURE_BLOB_CONTAINER in .env
  2. Replace PlaceholderBlobFetcher.fetch_image_as_base64() with the real
     BlobServiceClient call shown in the docstring below.
  3. The response-writer agent already accepts base64 images; no other
     changes needed.
"""
from __future__ import annotations

import base64
import logging

logger = logging.getLogger(__name__)

# 1×1 transparent PNG – used by the placeholder so the vision LLM won't error
_TRANSPARENT_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class PlaceholderBlobFetcher:
    """
    Returns a 1×1 transparent PNG stub.

    REPLACE this class body with:
    ────────────────────────────
    from azure.storage.blob import BlobServiceClient
    from config.settings import get_settings
    import base64

    class AzureBlobFetcher:
        def __init__(self):
            cfg = get_settings()
            self._client = BlobServiceClient.from_connection_string(
                cfg.azure_blob_connection_string
            )
            self._container = cfg.azure_blob_container

        def fetch_image_as_base64(self, blob_path: str) -> str:
            blob_client = self._client.get_blob_client(
                container=self._container, blob=blob_path
            )
            data = blob_client.download_blob().readall()
            return base64.b64encode(data).decode("utf-8")
    """

    def fetch_image_as_base64(self, blob_path: str) -> str:
        logger.info("[PLACEHOLDER BLOB] fetching blob_path=%r", blob_path)
        return _TRANSPARENT_PNG_B64


# ── Singleton factory ──────────────────────────────────────────────────────────

_blob_instance: PlaceholderBlobFetcher | None = None


def get_blob_fetcher() -> PlaceholderBlobFetcher:
    """
    Return the active blob fetcher.
    Swap PlaceholderBlobFetcher → AzureBlobFetcher here when ready.
    """
    global _blob_instance
    if _blob_instance is None:
        _blob_instance = PlaceholderBlobFetcher()
    return _blob_instance
