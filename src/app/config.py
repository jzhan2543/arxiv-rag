"""Settings from environment vars. Adapters read via Settings.from_env(), not os.environ."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    voyage_api_key: str
    index_path: Path

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            voyage_api_key=os.environ.get("VOYAGE_API_KEY", ""),
            index_path=Path(os.environ.get("INDEX_PATH", "data/index.db")),
        )
