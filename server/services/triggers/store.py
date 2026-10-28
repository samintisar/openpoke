from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ...logging_config import logger
from .models import TriggerRecord
from .utils import to_storage_timestamp, utc_now


class TriggerStore:
    """Low-level persistence for triggers backed by SQLite."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._ensure_directory()
        self._ensure_schema()

    def _ensure_directory(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "trigger directory creation failed",
                extra={"error": str(exc)},
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        schema_sql = """
        CREATE TABLE IF NOT EXISTS triggers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT,
            agent_name TEXT NOT NULL,
            payload TEXT NOT NULL,
            start_time TEXT,
            next_trigger TEXT,
            recurrence_rule TEXT,
            timezone TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
        index_sql = """
        CREATE INDEX IF NOT EXISTS idx_triggers_agent_id_next
        ON triggers (agent_id, next_trigger);
        """
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(schema_sql)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(triggers)")}
            if "agent_id" not in columns:
                # Rows from before agent ids are backfilled by name via `backfill_agent_ids`.
                conn.execute("ALTER TABLE triggers ADD COLUMN agent_id TEXT")
            conn.execute(index_sql)

    def insert(self, payload: Dict[str, Any]) -> int:
        with self._lock, self._connect() as conn:
            columns = ", ".join(payload.keys())
            placeholders = ", ".join([":" + key for key in payload.keys()])
            sql = f"INSERT INTO triggers ({columns}) VALUES ({placeholders})"
            conn.execute(sql, payload)
            trigger_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            return int(trigger_id)

    def fetch_one(self, trigger_id: int, agent_id: str) -> Optional[TriggerRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM triggers WHERE id = ? AND agent_id = ?",
                (trigger_id, agent_id),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def fetch_by_id(self, trigger_id: int) -> Optional[TriggerRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM triggers WHERE id = ?", (trigger_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def update(self, trigger_id: int, agent_id: str, fields: Dict[str, Any]) -> bool:
        if not fields:
            return False
        assignments = ", ".join(f"{key} = :{key}" for key in fields.keys())
        sql = (
            f"UPDATE triggers SET {assignments}, updated_at = :updated_at"
            " WHERE id = :trigger_id AND agent_id = :agent_id"
        )
        payload = {
            **fields,
            "updated_at": to_storage_timestamp(utc_now()),
            "trigger_id": trigger_id,
            "agent_id": agent_id,
        }
        with self._lock, self._connect() as conn:
            cursor = conn.execute(sql, payload)
            return cursor.rowcount > 0

    def delete(self, trigger_id: int, agent_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM triggers WHERE id = ? AND agent_id = ?", (trigger_id, agent_id))
            return cursor.rowcount > 0

    def list_for_agent(self, agent_id: str) -> List[TriggerRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM triggers WHERE agent_id = ? ORDER BY next_trigger IS NULL, next_trigger",
                (agent_id,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def fetch_due(
        self, agent_id: Optional[str], before_iso: str
    ) -> List[TriggerRecord]:
        sql = (
            "SELECT * FROM triggers WHERE status = 'active' AND next_trigger IS NOT NULL"
            " AND next_trigger <= ?"
        )
        params: List[Any] = [before_iso]
        if agent_id:
            sql += " AND agent_id = ?"
            params.append(agent_id)
        sql += " ORDER BY next_trigger, id"
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def backfill_agent_ids(self, resolve: Callable[[str], Optional[str]]) -> int:
        """Set `agent_id` on rows that only have a name; names `resolve` returns None for stay unset."""
        with self._lock, self._connect() as conn:
            names = [
                row[0]
                for row in conn.execute("SELECT DISTINCT agent_name FROM triggers WHERE agent_id IS NULL")
            ]
        updated = 0
        for name in names:
            agent_id = resolve(name)
            if not agent_id:
                logger.warning("trigger agent not found in registry", extra={"agent": name})
                continue
            with self._lock, self._connect() as conn:
                cursor = conn.execute(
                    "UPDATE triggers SET agent_id = ? WHERE agent_id IS NULL AND agent_name = ?",
                    (agent_id, name),
                )
                updated += cursor.rowcount
        return updated

    def clear_all(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM triggers")

    def _row_to_record(self, row: sqlite3.Row) -> TriggerRecord:
        data = dict(row)
        return TriggerRecord.model_validate(data)


__all__ = ["TriggerStore"]
