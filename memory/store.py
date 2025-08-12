"""Lightweight SQLite storage utility used by memory modules.

This module avoids external dependencies and provides a thin wrapper
around SQLite with sane defaults (WAL, foreign keys, row factory, thread safety).
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Optional, Sequence
from types import TracebackType


class SQLiteStore:
    """Thread-safe convenience wrapper around ``sqlite3.Connection``.

    Parameters
    - db_path: Path to the SQLite database file. Use ``":memory:"`` for in-memory.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False enables usage across threads guarded by self._lock
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._configure()

    def _configure(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")

    # Basic execution methods
    def execute(self, sql: str, params: Sequence | None = None) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params or [])
            self._conn.commit()
            return cur

    def executemany(
        self, sql: str, seq_of_params: Iterable[Sequence]
    ) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.executemany(sql, seq_of_params)
            self._conn.commit()
            return cur

    def query_all(self, sql: str, params: Sequence | None = None) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, params or [])
            return cur.fetchall()

    def query_one(
        self, sql: str, params: Sequence | None = None
    ) -> Optional[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, params or [])
            return cur.fetchone()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # Context manager support
    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
