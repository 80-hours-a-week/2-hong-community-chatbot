from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


@dataclass(frozen=True)
class Settings:
    raw_data_dir: Path = Path(os.getenv("RAW_DATA_DIR", "."))
    processed_data_dir: Path = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "auto")
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "5"))
    conversation_ttl_seconds: int = int(os.getenv("CONVERSATION_TTL_SECONDS", "3600"))


settings = Settings()
