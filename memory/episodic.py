from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .store import SQLiteStore


@dataclass(frozen=True)
class Episode:
    """Represents a time-bounded event for a user.

    - user_id: application-level user identifier
    - content: raw text content of the event
    - timestamp: ISO-8601 string (UTC recommended)
    - tags: optional comma-separated labels for filtering
    - source: origin of the event (e.g., chat, email, api)
    - metadata_json: JSON string of arbitrary attributes
    """

    user_id: str
    content: str
    timestamp: str
    tags: Optional[str] = None
    source: Optional[str] = None
    metadata_json: Optional[str] = None


class EpisodicMemory:
    """Manages storage and retrieval of episodic memories.

    The implementation uses ``SQLiteStore`` by default and auto-creates the schema.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.store = SQLiteStore(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.store.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                tags TEXT,
                source TEXT,
                metadata_json TEXT
            );
            """
        )
        self.store.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_episodes_user_time
            ON episodes (user_id, timestamp DESC);
            """
        )

    # Ingestion
    def add(self, episode: Episode) -> int:
        cur = self.store.execute(
            """
            INSERT INTO episodes (user_id, content, timestamp, tags, source, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                episode.user_id,
                episode.content,
                episode.timestamp,
                episode.tags,
                episode.source,
                episode.metadata_json,
            ],
        )
        last_id = cur.lastrowid
        return int(last_id) if last_id is not None else 0

    def add_many(self, episodes: Iterable[Episode]) -> None:
        params = [
            [e.user_id, e.content, e.timestamp, e.tags, e.source, e.metadata_json]
            for e in episodes
        ]
        if not params:
            return
        self.store.executemany(
            """
            INSERT INTO episodes (user_id, content, timestamp, tags, source, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            params,
        )

    # Retrieval
    def timeline(
        self, user_id: str, limit: int = 100, since: Optional[str] = None
    ) -> List[Episode]:
        if since is None:
            rows = self.store.query_all(
                """
                SELECT user_id, content, timestamp, tags, source, metadata_json
                FROM episodes
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                [user_id, limit],
            )
        else:
            rows = self.store.query_all(
                """
                SELECT user_id, content, timestamp, tags, source, metadata_json
                FROM episodes
                WHERE user_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                [user_id, since, limit],
            )
        return [Episode(**dict(row)) for row in rows]

    def search(self, user_id: str, query: str, limit: int = 50) -> List[Episode]:
        # Basic LIKE search; can be replaced with FTS5 later
        pattern = f"%{query}%"
        rows = self.store.query_all(
            """
            SELECT user_id, content, timestamp, tags, source, metadata_json
            FROM episodes
            WHERE user_id = ? AND content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [user_id, pattern, limit],
        )
        return [Episode(**dict(row)) for row in rows]

    # Maintenance
    def delete_before(self, user_id: str, before_timestamp: str) -> int:
        cur = self.store.execute(
            """
            DELETE FROM episodes
            WHERE user_id = ? AND timestamp < ?
            """,
            [user_id, before_timestamp],
        )
        return int(cur.rowcount or 0)

    def close(self) -> None:
        self.store.close()
