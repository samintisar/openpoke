"""Triggers are keyed by agent id; rows from before ids existed are backfilled by name."""

import sqlite3
from datetime import datetime, timezone

from server.services.triggers.service import TriggerService
from server.services.triggers.store import TriggerStore

FUTURE = "2099-01-01T09:00:00Z"


def service(tmp_path):
    return TriggerService(TriggerStore(tmp_path / "triggers.db"))


def test_lookup_is_by_agent_id(tmp_path):
    svc = service(tmp_path)
    # Two agents with the same display name (the F2 slug collision) stay separate.
    first = svc.create_trigger(agent_id="email-to-alice", agent_name="Email to Alice", payload="one", start_time=FUTURE)
    second = svc.create_trigger(agent_id="email-to-alice-2", agent_name="Email to Alice", payload="two", start_time=FUTURE)
    assert first.agent_id == "email-to-alice" and first.agent_name == "Email to Alice"
    assert [t.payload for t in svc.list_triggers(agent_id="email-to-alice")] == ["one"]
    assert [t.payload for t in svc.list_triggers(agent_id="email-to-alice-2")] == ["two"]

    assert svc.update_trigger(second.id, agent_id="email-to-alice", payload="hijack") is None
    updated = svc.update_trigger(second.id, agent_id="email-to-alice-2", status="paused")
    assert updated.status == "paused"
    due = svc.get_due_triggers(before=datetime(2100, 1, 1, tzinfo=timezone.utc), agent_id="email-to-alice")
    assert [t.id for t in due] == [first.id]


def test_old_rows_get_agent_id_column_and_backfill(tmp_path):
    path = tmp_path / "triggers.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE triggers (id INTEGER PRIMARY KEY AUTOINCREMENT, agent_name TEXT NOT NULL, payload TEXT NOT NULL,"
        " start_time TEXT, next_trigger TEXT, recurrence_rule TEXT, timezone TEXT, status TEXT NOT NULL DEFAULT 'active',"
        " last_error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    for name in ("Email to Alice", "Ghost Agent"):
        conn.execute(
            "INSERT INTO triggers (agent_name, payload, next_trigger, created_at, updated_at) VALUES (?, 'p', ?, 't', 't')",
            (name, FUTURE),
        )
    conn.commit()
    conn.close()

    svc = TriggerService(TriggerStore(path))
    assert svc.list_triggers(agent_id="email-to-alice") == []
    resolved = {"Email to Alice": "email-to-alice"}
    assert svc.backfill_agent_ids(resolved.get) == 1
    rows = svc.list_triggers(agent_id="email-to-alice")
    assert [(t.agent_id, t.agent_name) for t in rows] == [("email-to-alice", "Email to Alice")]
    # Unresolved names stay unset until a resolver knows them; backfilling again is idempotent.
    assert svc.backfill_agent_ids(lambda name: "ghost-agent") == 1
    assert svc.backfill_agent_ids(lambda name: "other") == 0
    assert [t.payload for t in svc.list_triggers(agent_id="ghost-agent")] == ["p"]
