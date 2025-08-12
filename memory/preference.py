from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .store import SQLiteStore


@dataclass(frozen=True)
class Preference:
    user_id: str
    key: str
    value_json: str
    confidence: float
    source_episode_id: Optional[int]


class PreferenceStore:
    """Simple key/value JSON preference storage with last-writer-wins semantics."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.store = SQLiteStore(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.store.execute(
            """
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_episode_id INTEGER,
                last_updated TEXT NOT NULL,
                UNIQUE(user_id, key)
            );
            """
        )
        self.store.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_preferences_user ON preferences(user_id);
            """
        )

    def upsert(self, pref: Preference, last_updated: str) -> None:
        self.store.execute(
            """
            INSERT INTO preferences (user_id, key, value_json, confidence, source_episode_id, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET
                value_json=excluded.value_json,
                confidence=excluded.confidence,
                source_episode_id=excluded.source_episode_id,
                last_updated=excluded.last_updated
            ;
            """,
            [
                pref.user_id,
                pref.key,
                pref.value_json,
                pref.confidence,
                pref.source_episode_id,
                last_updated,
            ],
        )

    def get_all(self, user_id: str) -> List[Dict[str, Any]]:
        rows = self.store.query_all(
            """
            SELECT key, value_json, confidence, source_episode_id, last_updated
            FROM preferences
            WHERE user_id = ?
            ORDER BY key ASC
            """,
            [user_id],
        )
        return [dict(row) for row in rows]

    def clear(self, user_id: str) -> int:
        cur = self.store.execute(
            """
            DELETE FROM preferences WHERE user_id = ?
            """,
            [user_id],
        )
        return int(cur.rowcount or 0)

    def close(self) -> None:
        self.store.close()
