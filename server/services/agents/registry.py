"""SQLite-backed agent registry: the production `AgentDirectory` (replaces `roster.json`)."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from ...logging_config import logger
from .cards import AgentCard, as_utc, extract_entities, normalize_name, slug_id
from .directory import merge_entities

STATUS_CHARS = 200
_LAST_ROUTED_KEY = "last_routed"


def _to_storage(moment: datetime) -> str:
    return as_utc(moment).isoformat()


def _from_storage(value: str) -> datetime:
    return as_utc(datetime.fromisoformat(value))


class SqliteAgentDirectory:
    """`AgentDirectory` over an `agents` table, with `last_routed` kept in a `meta` table."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._ensure_directory()
        self._ensure_schema()

    def _ensure_directory(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("agent registry directory creation failed", extra={"error": str(exc)})

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        agents_sql = """
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            purpose TEXT NOT NULL,
            entities TEXT NOT NULL DEFAULT '[]',
            last_status TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            last_active_at TEXT NOT NULL,
            run_count INTEGER NOT NULL DEFAULT 0
        );
        """
        meta_sql = """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(agents_sql)
            conn.execute(meta_sql)

    # -- AgentDirectory -------------------------------------------------------------------------------

    def list_cards(self) -> list[AgentCard]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM agents ORDER BY created_at, rowid").fetchall()
        return [self._row_to_card(row) for row in rows]

    def get(self, agent_id: str) -> Optional[AgentCard]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return self._row_to_card(row) if row else None

    def create(self, name: str, purpose: str, instructions: str, now: datetime) -> AgentCard:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                taken = [row[0] for row in conn.execute("SELECT id FROM agents")]
                card = AgentCard(
                    id=slug_id(name, taken),
                    name=name,
                    purpose=purpose,
                    entities=extract_entities([name, purpose, instructions]),
                    created_at=as_utc(now),
                    last_active_at=as_utc(now),
                    run_count=1,
                )
                self._insert(conn, card)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return card

    def touch(self, agent_id: str, instructions: str, now: datetime) -> AgentCard:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if row is None:
                raise KeyError(agent_id)
            card = self._row_to_card(row)
            card.entities = merge_entities(card.entities, [instructions])
            card.last_active_at = as_utc(now)
            card.run_count += 1
            card.status = "active"
            conn.execute(
                "UPDATE agents SET entities = ?, last_active_at = ?, run_count = ?, status = ? WHERE id = ?",
                (json.dumps(card.entities), _to_storage(card.last_active_at), card.run_count, card.status, agent_id),
            )
        return card

    def last_routed(self) -> list[str]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (_LAST_ROUTED_KEY,)).fetchone()
        if row is None:
            return []
        try:
            value = json.loads(row[0])
        except json.JSONDecodeError:
            return []
        return [str(item) for item in value] if isinstance(value, list) else []

    def set_last_routed(self, ids: Iterable[str]) -> None:
        payload = json.dumps(list(dict.fromkeys(ids)))
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_LAST_ROUTED_KEY, payload),
            )

    # -- Registry extras ------------------------------------------------------------------------------

    def record_status(self, agent_id: str, text: str) -> None:
        """Keep the first `STATUS_CHARS` characters of the agent's latest final response."""
        status = (text or "").strip()[:STATUS_CHARS]
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE agents SET last_status = ? WHERE id = ?", (status, agent_id))

    def update_card(
        self, agent_id: str, *, purpose: Optional[str] = None, last_status: Optional[str] = None,
        status: Optional[str] = None,
    ) -> None:
        """Overwrite the given fields (compaction refreshes purpose and status; the sweep sets lifecycle)."""
        fields = {"purpose": purpose, "last_status": None if last_status is None else last_status.strip()[:STATUS_CHARS],
                  "status": status}
        updates = {k: v for k, v in fields.items() if v is not None}
        if not updates:
            return
        assignments = ", ".join(f"{column} = ?" for column in updates)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE agents SET {assignments} WHERE id = ?", (*updates.values(), agent_id))

    def find_by_name(self, name: str) -> Optional[AgentCard]:
        """Exact display-name match first, then normalized; oldest card wins."""
        cards = self.list_cards()
        exact = next((card for card in cards if card.name == name), None)
        if exact is not None:
            return exact
        wanted = normalize_name(name)
        return next((card for card in cards if normalize_name(card.name) == wanted), None)

    def add_cards(self, cards: Iterable[AgentCard]) -> None:
        """Insert fully formed cards in one transaction (used by the roster migration)."""
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for card in cards:
                    self._insert(conn, card)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def is_empty(self) -> bool:
        with self._lock, self._connect() as conn:
            return conn.execute("SELECT 1 FROM agents LIMIT 1").fetchone() is None

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM agents")
            conn.execute("DELETE FROM meta")

    # -- Internals ------------------------------------------------------------------------------------

    def _insert(self, conn: sqlite3.Connection, card: AgentCard) -> None:
        conn.execute(
            "INSERT INTO agents (id, name, purpose, entities, last_status, status, created_at, last_active_at,"
            " run_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                card.id,
                card.name,
                card.purpose,
                json.dumps(list(card.entities)),
                card.last_status,
                card.status,
                _to_storage(card.created_at),
                _to_storage(card.last_active_at),
                card.run_count,
            ),
        )

    def _row_to_card(self, row: sqlite3.Row) -> AgentCard:
        try:
            entities = json.loads(row["entities"] or "[]")
        except json.JSONDecodeError:
            entities = []
        return AgentCard(
            id=row["id"],
            name=row["name"],
            purpose=row["purpose"],
            entities=[str(e) for e in entities] if isinstance(entities, list) else [],
            last_status=row["last_status"] or "",
            status=row["status"] or "active",
            created_at=_from_storage(row["created_at"]),
            last_active_at=_from_storage(row["last_active_at"]),
            run_count=int(row["run_count"] or 0),
        )


def _parse_log_timestamp(value: str, tz: Any) -> Optional[datetime]:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=tz).astimezone(timezone.utc)


def migrate_roster(
    directory: SqliteAgentDirectory,
    roster_path: Path,
    log_store: Any,
    now: datetime,
    log_timezone: Any = timezone.utc,
) -> int:
    """Import `roster.json` into an empty registry, then rename it to `roster.json.migrated`.

    Each name becomes a card with `purpose = name` and entities extracted from its logged requests. Its log
    (keyed by the old name slug) is renamed to the new card id. Returns the number of cards imported.
    """
    if not roster_path.exists() or not directory.is_empty():
        return 0
    try:
        data = json.loads(roster_path.read_text(encoding="utf-8") or "[]")
    except Exception as exc:
        logger.warning("roster.json unreadable; skipping migration", extra={"error": str(exc)})
        return 0
    names = list(dict.fromkeys(str(name) for name in data)) if isinstance(data, list) else []

    cards: list[AgentCard] = []
    renames: list[tuple[str, str]] = []
    for name in names:
        entries = list(log_store.iter_entries(name))
        requests = [payload for tag, _, payload in entries if tag == "agent_request"]
        stamps = [ts for ts in (_parse_log_timestamp(stamp, log_timezone) for _, stamp, _ in entries) if ts]
        card_id = slug_id(name, [card.id for card in cards])
        cards.append(
            AgentCard(
                id=card_id,
                name=name,
                purpose=name,
                entities=extract_entities([name, *requests]),
                created_at=min(stamps) if stamps else as_utc(now),
                last_active_at=max(stamps) if stamps else as_utc(now),
                run_count=len(requests),
            )
        )
        renames.append((name, card_id))

    directory.add_cards(cards)
    for old_key, new_key in renames:
        log_store.rename_agent(old_key, new_key)
    try:
        roster_path.replace(roster_path.with_name(roster_path.name + ".migrated"))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("failed to rename roster.json after migration", extra={"error": str(exc)})
    logger.info(f"Migrated {len(cards)} agents from roster.json")
    return len(cards)


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DB_PATH = _DATA_DIR / "agents.db"
_ROSTER_PATH = _DATA_DIR / "execution_agents" / "roster.json"

_directory: Optional[SqliteAgentDirectory] = None
_directory_lock = threading.Lock()


def _initialize(directory: SqliteAgentDirectory) -> None:
    from ..execution import get_execution_agent_logs
    from ..execution.log_store import _slugify
    from ..triggers import get_trigger_service
    from ...utils.timezones import resolve_user_timezone

    try:
        migrate_roster(
            directory,
            _ROSTER_PATH,
            get_execution_agent_logs(),
            datetime.now(timezone.utc),
            resolve_user_timezone(),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("roster.json migration failed", extra={"error": str(exc)})
    def resolve(name: str) -> str:
        # A name the registry doesn't know keeps its old log key, so its trigger still fires against its log.
        card = directory.find_by_name(name)
        return card.id if card else _slugify(name)

    try:
        get_trigger_service().backfill_agent_ids(resolve)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("trigger agent_id backfill failed", extra={"error": str(exc)})


def get_agent_directory() -> SqliteAgentDirectory:
    """The registry singleton at `server/data/agents.db`; migrates `roster.json` on first use."""
    global _directory
    with _directory_lock:
        if _directory is None:
            directory = SqliteAgentDirectory(_DB_PATH)
            _initialize(directory)
            _directory = directory
        return _directory


__all__ = ["STATUS_CHARS", "SqliteAgentDirectory", "get_agent_directory", "migrate_roster"]
