from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from ...logging_config import logger

KINDS = ("user", "agent_result", "trigger_result", "watcher")
STATUSES = ("pending", "processing", "done")


@dataclass(frozen=True)
class InboxItem:
    id: int
    kind: str
    payload: str
    agent_id: Optional[str]
    created_at: datetime  # timezone-aware UTC
    status: str


def _to_storage(moment: datetime) -> str:
    """UTC ISO 8601 with microseconds, so user-message times compare exactly against approval times."""
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


class InboxStore:
    """Persistence for interaction-inbox items backed by SQLite."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._ensure_directory()
        self._ensure_schema()

    def _ensure_directory(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("inbox directory creation failed", extra={"error": str(exc)})

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        schema_sql = """
        CREATE TABLE IF NOT EXISTS inbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            agent_id TEXT,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
        );
        """
        index_sql = "CREATE INDEX IF NOT EXISTS idx_inbox_status_id ON inbox (status, id);"
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(schema_sql)
            conn.execute(index_sql)

    def insert(
        self, kind: str, payload: str, agent_id: Optional[str] = None, *, created_at: Optional[datetime] = None
    ) -> InboxItem:
        if kind not in KINDS:
            raise ValueError(f"Unknown inbox kind: {kind!r}")
        moment = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO inbox (kind, payload, agent_id, created_at, status) VALUES (?, ?, ?, ?, 'pending')",
                (kind, payload, agent_id, _to_storage(moment)),
            )
            item_id = int(cursor.lastrowid)
        return InboxItem(item_id, kind, payload, agent_id, moment, "pending")

    def pending(self) -> List[InboxItem]:
        """Pending items, oldest first."""
        return self._select("status = 'pending'")

    def unfinished(self) -> List[InboxItem]:
        """Items not yet done (pending or processing), oldest first."""
        return self._select("status != 'done'")

    def mark(self, ids: Iterable[int], status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"Unknown inbox status: {status!r}")
        id_list = [int(item_id) for item_id in ids]
        if not id_list:
            return
        placeholders = ", ".join("?" for _ in id_list)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE inbox SET status = ? WHERE id IN ({placeholders})", (status, *id_list))

    def reset_processing(self) -> int:
        """Return items a stopped consumer left mid-turn to `pending`; the count reset."""
        with self._lock, self._connect() as conn:
            cursor = conn.execute("UPDATE inbox SET status = 'pending' WHERE status = 'processing'")
            return cursor.rowcount

    def clear_all(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM inbox")

    def _select(self, where: str) -> List[InboxItem]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM inbox WHERE {where} ORDER BY id").fetchall()
        return [self._row_to_item(row) for row in rows]

    def _row_to_item(self, row: sqlite3.Row) -> InboxItem:
        created_at = datetime.fromisoformat(row["created_at"])
        if created_at.tzinfo is None:  # pragma: no cover - rows are always written with an offset
            created_at = created_at.replace(tzinfo=timezone.utc)
        return InboxItem(row["id"], row["kind"], row["payload"], row["agent_id"], created_at, row["status"])


__all__ = ["InboxItem", "InboxStore", "KINDS"]
