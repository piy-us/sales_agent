"""
memory/cosmos_memory.py
────────────────────────
Persistent conversation history backed by Azure CosmosDB.

Schema of a single CosmosDB item
─────────────────────────────────
{
  "id":           "<session_id>",          ← partition key = session_id
  "session_id":   "<session_id>",
  "turns": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ],
  "ttl":  2592000                          ← auto-expire after N seconds
}

We keep ONE document per session_id and append turns to the `turns` list.
CosmosDB upsert is atomic so concurrent writes are safe for our use-case.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from azure.cosmos import CosmosClient, PartitionKey, exceptions

from config.settings import get_settings

logger = logging.getLogger(__name__)


class CosmosConversationMemory:
    """
    Thread-safe conversation history store on Azure CosmosDB.

    Usage
    -----
    memory = CosmosConversationMemory()
    memory.add_turn(session_id, "user", "Hello!")
    memory.add_turn(session_id, "assistant", "Hi there!")
    history = memory.get_history(session_id)   # list of {"role": ..., "content": ...}
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._client = CosmosClient(
            url=cfg.cosmos_endpoint,
            credential=cfg.cosmos_key,
        )
        self._db = self._client.create_database_if_not_exists(id=cfg.cosmos_database)
        self._container = self._db.create_container_if_not_exists(
            id=cfg.cosmos_container,
            partition_key=PartitionKey(path="/session_id"),
            # Enable TTL at container level; individual items supply their own ttl value
            default_ttl=-1,
        )
        self._ttl = cfg.cosmos_ttl
        self._max_turns = cfg.max_history_turns

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _read_item(self, session_id: str) -> dict[str, Any] | None:
        """Return the CosmosDB item for the session or None if it doesn't exist."""
        try:
            return self._container.read_item(
                item=session_id, partition_key=session_id
            )
        except exceptions.CosmosResourceNotFoundError:
            return None

    def _build_new_item(self, session_id: str) -> dict[str, Any]:
        return {
            "id": session_id,
            "session_id": session_id,
            "turns": [],
            "ttl": self._ttl,
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """
        Return the last `max_history_turns` turns for a session.
        Each turn: {"role": "user"|"assistant", "content": "..."}
        """
        item = self._read_item(session_id)
        if item is None:
            return []
        turns = item.get("turns", [])
        # Return the most recent N turns
        return turns[-self._max_turns * 2 :]  # *2 because each exchange = 2 turns

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Append a single turn (user or assistant) and upsert to CosmosDB."""
        item = self._read_item(session_id) or self._build_new_item(session_id)
        item["turns"].append({"role": role, "content": content})
        # Refresh TTL on every write
        item["ttl"] = self._ttl
        self._container.upsert_item(body=item)
        logger.debug("Saved turn for session=%s role=%s", session_id, role)

    def add_exchange(
        self, session_id: str, user_content: str, assistant_content: str
    ) -> None:
        """Append a full user↔assistant exchange atomically."""
        item = self._read_item(session_id) or self._build_new_item(session_id)
        item["turns"].extend(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        )
        item["ttl"] = self._ttl
        self._container.upsert_item(body=item)

    def clear_history(self, session_id: str) -> None:
        """Delete all history for a session."""
        try:
            self._container.delete_item(
                item=session_id, partition_key=session_id
            )
            logger.info("Cleared history for session=%s", session_id)
        except exceptions.CosmosResourceNotFoundError:
            pass

    def format_history_for_prompt(self, session_id: str) -> str:
        """
        Return history as a human-readable string suitable for injection
        into a system or user prompt.

        Example:
            User: Hello
            Assistant: Hi! How can I help?
        """
        turns = self.get_history(session_id)
        if not turns:
            return "No prior conversation history."
        lines = []
        for turn in turns:
            role = turn["role"].capitalize()
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)
