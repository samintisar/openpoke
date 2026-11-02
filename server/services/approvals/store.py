from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...logging_config import logger

KINDS = ("draft", "reply", "forward")
STATUSES = ("pending", "approved", "sent", "expired", "invalidated")


@dataclass(frozen=True)
class PendingAction:
    """One send waiting for (or past) the user's approval; the content is what the user is shown and approves."""

    id: int
    kind: str
    agent_id: Optional[str]
    gmail_ref: Optional[str]  # draft id, thread id (reply) or message id (forward)
    recipients: Dict[str, List[str]]  # {"to": [...], "cc": [...], "bcc": [...]}, normalized
    subject: str
    body: str  # the body, or the forward's additional text
    body_format: str  # "plain" or "html"
    content_hash: str
    shown_at: Optional[datetime]
    approved_at: Optional[datetime]
    status: str
    created_at: datetime  # timezone-aware UTC


@dataclass(frozen=True)
class StandingApproval:
    """The user's standing yes for one trigger's sends to a fixed set of recipients, until revoked."""

    id: int
    trigger_id: int
    agent_id: str
    recipients: List[str]  # normalized, sorted
    created_at: datetime  # timezone-aware UTC
    revoked_at: Optional[datetime]


def to_storage(moment: datetime) -> str:
    """UTC ISO 8601 with microseconds, so shown/approval times compare exactly against user-message times."""
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _from_storage(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:  # pragma: no cover - rows are always written with an offset
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


class PendingActionStore:
    """Persistence for send approvals (`pending_actions`) backed by SQLite."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._ensure_directory()
        self._ensure_schema()

    def _ensure_directory(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("approvals directory creation failed", extra={"error": str(exc)})

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        schema_sql = """
        CREATE TABLE IF NOT EXISTS pending_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            agent_id TEXT,
            gmail_ref TEXT,
            recipients TEXT NOT NULL,
            subject TEXT NOT NULL DEFAULT '',
            body TEXT NOT NULL DEFAULT '',
            body_format TEXT NOT NULL DEFAULT 'plain',
            content_hash TEXT NOT NULL,
            shown_at TEXT,
            approved_at TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
        );
        """
        index_sql = """
        CREATE INDEX IF NOT EXISTS idx_pending_actions_lookup
        ON pending_actions (kind, gmail_ref, agent_id, status);
        """
        standing_sql = """
        CREATE TABLE IF NOT EXISTS standing_approvals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger_id INTEGER NOT NULL,
            agent_id TEXT NOT NULL,
            recipients TEXT NOT NULL,
            created_at TEXT NOT NULL,
            revoked_at TEXT
        );
        """
        standing_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_standing_approvals_trigger
        ON standing_approvals (trigger_id, agent_id, revoked_at);
        """
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(schema_sql)
            conn.execute(index_sql)
            conn.execute(standing_sql)
            conn.execute(standing_index_sql)

    def insert(
        self,
        *,
        kind: str,
        agent_id: Optional[str],
        gmail_ref: Optional[str],
        recipients: Dict[str, List[str]],
        subject: str,
        body: str,
        body_format: str,
        content_hash: str,
        created_at: datetime,
    ) -> PendingAction:
        if kind not in KINDS:
            raise ValueError(f"Unknown action kind: {kind!r}")
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO pending_actions (kind, agent_id, gmail_ref, recipients, subject, body, body_format,"
                " content_hash, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (
                    kind,
                    agent_id,
                    gmail_ref,
                    json.dumps(recipients, sort_keys=True),
                    subject,
                    body,
                    body_format,
                    content_hash,
                    to_storage(created_at),
                ),
            )
            action_id = int(cursor.lastrowid)
            row = conn.execute("SELECT * FROM pending_actions WHERE id = ?", (action_id,)).fetchone()
        return self._row_to_action(row)

    def get(self, action_id: int) -> Optional[PendingAction]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM pending_actions WHERE id = ?", (int(action_id),)).fetchone()
        return self._row_to_action(row) if row else None

    def find(
        self,
        *,
        kind: str,
        gmail_ref: Optional[str],
        agent_id: Optional[str] = None,
        statuses: Tuple[str, ...] = (),
    ) -> List[PendingAction]:
        """Rows for one Gmail object (and agent, when given), newest first."""
        sql = "SELECT * FROM pending_actions WHERE kind = ? AND gmail_ref IS ?"
        params: List[Any] = [kind, gmail_ref]
        if agent_id is not None:
            sql += " AND agent_id = ?"
            params.append(agent_id)
        if statuses:
            sql += f" AND status IN ({', '.join('?' for _ in statuses)})"
            params.extend(statuses)
        sql += " ORDER BY id DESC"
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_action(row) for row in rows]

    def with_status(self, statuses: Tuple[str, ...]) -> List[PendingAction]:
        placeholders = ", ".join("?" for _ in statuses)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM pending_actions WHERE status IN ({placeholders}) ORDER BY id", statuses
            ).fetchall()
        return [self._row_to_action(row) for row in rows]

    def transition(
        self, action_id: int, from_statuses: Tuple[str, ...], to_status: str, **fields: Optional[datetime]
    ) -> bool:
        """Move a row to `to_status` only if it is in one of `from_statuses`; True when it moved."""
        if to_status not in STATUSES:
            raise ValueError(f"Unknown action status: {to_status!r}")
        assignments = ["status = ?"]
        params: List[Any] = [to_status]
        for key, value in fields.items():
            if key not in {"shown_at", "approved_at"}:
                raise ValueError(f"Unknown field: {key!r}")
            assignments.append(f"{key} = ?")
            params.append(to_storage(value) if value is not None else None)
        params.append(int(action_id))
        params.extend(from_statuses)
        sql = (
            f"UPDATE pending_actions SET {', '.join(assignments)} WHERE id = ?"
            f" AND status IN ({', '.join('?' for _ in from_statuses)})"
        )
        with self._lock, self._connect() as conn:
            return conn.execute(sql, params).rowcount > 0

    def set_shown(self, action_id: int, moment: datetime) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "UPDATE pending_actions SET shown_at = ? WHERE id = ? AND status = 'pending'",
                (to_storage(moment), int(action_id)),
            )
            return cursor.rowcount > 0

    def expire_older_than(self, cutoff: datetime, statuses: Tuple[str, ...]) -> int:
        placeholders = ", ".join("?" for _ in statuses)
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE pending_actions SET status = 'expired' WHERE status IN ({placeholders})"
                " AND created_at < ?",
                (*statuses, to_storage(cutoff)),
            )
            return cursor.rowcount

    # -- standing approvals ----------------------------------------------------------------------------------

    def insert_standing(
        self, *, trigger_id: int, agent_id: str, recipients: List[str], created_at: datetime
    ) -> StandingApproval:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO standing_approvals (trigger_id, agent_id, recipients, created_at) VALUES (?, ?, ?, ?)",
                (int(trigger_id), agent_id, json.dumps(sorted(recipients)), to_storage(created_at)),
            )
            row = conn.execute(
                "SELECT * FROM standing_approvals WHERE id = ?", (int(cursor.lastrowid),)
            ).fetchone()
        return self._row_to_standing(row)

    def active_standing(self, trigger_id: int, agent_id: str) -> List[StandingApproval]:
        """Unrevoked standing approvals for this trigger and agent, oldest first."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM standing_approvals WHERE trigger_id = ? AND agent_id = ? AND revoked_at IS NULL"
                " ORDER BY id",
                (int(trigger_id), agent_id),
            ).fetchall()
        return [self._row_to_standing(row) for row in rows]

    def revoke_standing(self, moment: datetime, trigger_id: Optional[int] = None) -> int:
        """Revoke the unrevoked standing approvals of one trigger (or of every trigger); the count revoked."""
        sql = "UPDATE standing_approvals SET revoked_at = ? WHERE revoked_at IS NULL"
        params: List[Any] = [to_storage(moment)]
        if trigger_id is not None:
            sql += " AND trigger_id = ?"
            params.append(int(trigger_id))
        with self._lock, self._connect() as conn:
            return conn.execute(sql, params).rowcount

    def clear_all(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM pending_actions")
            conn.execute("DELETE FROM standing_approvals")

    def _row_to_standing(self, row: sqlite3.Row) -> StandingApproval:
        return StandingApproval(
            id=int(row["id"]),
            trigger_id=int(row["trigger_id"]),
            agent_id=row["agent_id"],
            recipients=json.loads(row["recipients"]),
            created_at=_from_storage(row["created_at"]),  # type: ignore[arg-type]
            revoked_at=_from_storage(row["revoked_at"]),
        )

    def _row_to_action(self, row: sqlite3.Row) -> PendingAction:
        return PendingAction(
            id=int(row["id"]),
            kind=row["kind"],
            agent_id=row["agent_id"],
            gmail_ref=row["gmail_ref"],
            recipients=json.loads(row["recipients"]),
            subject=row["subject"] or "",
            body=row["body"] or "",
            body_format=row["body_format"] or "plain",
            content_hash=row["content_hash"],
            shown_at=_from_storage(row["shown_at"]),
            approved_at=_from_storage(row["approved_at"]),
            status=row["status"],
            created_at=_from_storage(row["created_at"]),  # type: ignore[arg-type]
        )


__all__ = ["KINDS", "PendingAction", "PendingActionStore", "STATUSES", "StandingApproval", "to_storage"]
