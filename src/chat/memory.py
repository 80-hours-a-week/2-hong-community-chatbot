from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any


class ConversationMemory:
    def __init__(self, max_turns: int = 5, ttl_seconds: int = 3600):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self.store = defaultdict(lambda: deque(maxlen=max_turns))
        self.updated_at: dict[str, float] = {}

    def get(self, conversation_id: str) -> list[dict[str, Any]]:
        self._expire_old()
        return list(self.store.get(conversation_id, []))

    def append(
        self,
        conversation_id: str,
        user_message: str,
        assistant_answer: str,
        recommendations: list[dict[str, Any]],
        parsed_query: dict[str, Any],
    ) -> None:
        self.store[conversation_id].append(
            {
                "user": user_message,
                "assistant": assistant_answer,
                "parsed_query": parsed_query,
                "recommendations": recommendations,
            }
        )
        self.updated_at[conversation_id] = time.time()

    def delete(self, conversation_id: str) -> None:
        self.store.pop(conversation_id, None)
        self.updated_at.pop(conversation_id, None)

    def _expire_old(self) -> None:
        now = time.time()
        expired = [
            key
            for key, updated in self.updated_at.items()
            if now - updated > self.ttl_seconds
        ]
        for key in expired:
            self.delete(key)
