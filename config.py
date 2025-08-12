from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    """Runtime configuration for Cortex.

    Only a minimal subset needed for the ingestion pipeline is implemented.
    """

    db_path: str = "./cortex.db"
    max_episodic_messages: int = 200
    pii_mode: str = "mask"  # "store" | "mask" | "ignore"

    @staticmethod
    def from_env(
        default_db_path: Optional[str] = None,
        default_max_episodes: Optional[int] = None,
    ) -> "Config":
        db_path = os.getenv("CORTEX_DB_PATH", default_db_path or "./cortex.db")
        try:
            max_episodes = int(
                os.getenv("MAX_EPISODIC_MESSAGES", default_max_episodes or 200)
            )
        except ValueError:
            max_episodes = default_max_episodes or 200
        pii_mode = os.getenv("PII_MODE", "mask")
        return Config(
            db_path=db_path, max_episodic_messages=max_episodes, pii_mode=pii_mode
        )
