"""SQLite agent registry: CRUD, id collisions, last_routed, record_status and the roster.json migration."""

import json
from datetime import datetime, timedelta, timezone

from server.services.agents import AgentCard, dispatch
from server.services.agents.registry import STATUS_CHARS, SqliteAgentDirectory, migrate_roster
from server.services.execution.log_store import ExecutionAgentLogStore

NOW = datetime(2026, 6, 26, 12, 0, tzinfo=timezone.utc)


def registry(tmp_path):
    return SqliteAgentDirectory(tmp_path / "agents.db")


def test_create_get_list_round_trip(tmp_path):
    d = registry(tmp_path)
    card = d.create("Gary Whitman Lease", "Renew the lease with Gary Whitman", "Email gary@example.com", NOW)
    assert card.id == "gary-whitman-lease"
    assert {"gary@example.com", "gary", "whitman"} <= set(card.entities)
    assert (card.run_count, card.status, card.created_at, card.last_active_at) == (1, "active", NOW, NOW)

    again = SqliteAgentDirectory(tmp_path / "agents.db").get("gary-whitman-lease")
    assert again == card
    assert [c.id for c in d.list_cards()] == ["gary-whitman-lease"]
    assert d.get("nope") is None


def test_id_collisions_get_suffixes(tmp_path):
    d = registry(tmp_path)
    ids = [d.create(name, "p", "", NOW).id for name in ("Email to Alice", "Email to Alice", "email-to alice!")]
    assert ids == ["email-to-alice", "email-to-alice-2", "email-to-alice-3"]
    assert [c.name for c in d.list_cards()] == ["Email to Alice", "Email to Alice", "email-to alice!"]


def test_touch_merges_entities_and_revives(tmp_path):
    d = registry(tmp_path)
    d.add_cards([AgentCard("kwame", "Kwame Dates", "p", ["kwame"], status="dormant", created_at=NOW, last_active_at=NOW)])
    later = NOW + timedelta(hours=2)
    card = d.touch("kwame", "Also loop in Sofia Reyes", later)
    assert card == d.get("kwame")
    assert card.entities == ["kwame", "sofia", "reyes"]
    assert (card.run_count, card.status, card.last_active_at) == (1, "active", later)


def test_dispatch_over_registry(tmp_path):
    d = registry(tmp_path)
    created = dispatch(d, instructions="Ask Kwame about dates", name="Kwame Book Club", purpose="Book club dates with Kwame", now=NOW)
    assert created.action == "created"
    reused = dispatch(d, instructions="Any reply?", agent_id=created.card.id, now=NOW)
    assert reused.action == "reused" and d.get(created.card.id).run_count == 2


def test_last_routed_persists(tmp_path):
    d = registry(tmp_path)
    assert d.last_routed() == []
    d.set_last_routed(["a", "b", "a"])
    assert SqliteAgentDirectory(tmp_path / "agents.db").last_routed() == ["a", "b"]
    d.set_last_routed([])
    assert d.last_routed() == []


def test_record_status_keeps_first_200_chars(tmp_path):
    d = registry(tmp_path)
    card = d.create("Alice", "p", "", NOW)
    d.record_status(card.id, "  " + "x" * 300)
    assert d.get(card.id).last_status == "x" * STATUS_CHARS
    d.record_status("unknown", "ignored")  # no row, no error


def test_find_by_name_and_clear(tmp_path):
    d = registry(tmp_path)
    first = d.create("Email to Alice", "p", "", NOW)
    d.create("Email to Alice", "p", "", NOW)
    assert d.find_by_name("Email to Alice").id == first.id
    assert d.find_by_name("email to alice!").id == first.id
    assert d.find_by_name("Bob") is None
    d.set_last_routed([first.id])
    d.clear()
    assert d.list_cards() == [] and d.last_routed() == [] and d.is_empty()


def test_migrates_roster_json(tmp_path):
    logs = ExecutionAgentLogStore(tmp_path / "execution_agents")
    roster = tmp_path / "execution_agents" / "roster.json"
    roster.write_text(json.dumps(["Gary's Lease", "Email to Alice", "Email to Alice"]), encoding="utf-8")
    old_log = tmp_path / "execution_agents" / "gary-s-lease.log"
    old_log.write_text(
        '<agent_request timestamp="2026-06-20 09:00:00">Call Gary Whitman about the lease</agent_request>\n'
        '<agent_response timestamp="2026-06-20 09:01:00">Done</agent_response>\n'
        '<agent_request timestamp="2026-06-21 10:00:00">Loop in Priya</agent_request>\n',
        encoding="utf-8",
    )
    d = registry(tmp_path)

    assert migrate_roster(d, roster, logs, NOW) == 2
    assert not roster.exists() and roster.with_name("roster.json.migrated").exists()
    gary, alice = d.list_cards()
    assert (gary.id, gary.name, gary.purpose) == ("garys-lease", "Gary's Lease", "Gary's Lease")
    assert {"gary", "whitman", "priya"} <= set(gary.entities)  # harvested from logged requests
    assert gary.run_count == 2
    assert gary.created_at == datetime(2026, 6, 20, 9, 0, tzinfo=timezone.utc)
    assert gary.last_active_at == datetime(2026, 6, 21, 10, 0, tzinfo=timezone.utc)
    # The log moved to the card id, so the agent keeps its history.
    assert not old_log.exists() and "Loop in Priya" in logs.load_transcript("garys-lease")
    assert (alice.id, alice.run_count, alice.last_active_at) == ("email-to-alice", 0, NOW)


def test_migration_skips_non_empty_registry(tmp_path):
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps(["Bob"]), encoding="utf-8")
    d = registry(tmp_path)
    d.create("Alice", "p", "", NOW)
    assert migrate_roster(d, roster, logs, NOW) == 0
    assert roster.exists() and [c.name for c in d.list_cards()] == ["Alice"]


def test_migration_without_roster_is_noop(tmp_path):
    d = registry(tmp_path)
    assert migrate_roster(d, tmp_path / "roster.json", ExecutionAgentLogStore(tmp_path / "logs"), NOW) == 0
    assert d.is_empty()


def test_initialize_migrates_and_backfills_triggers(tmp_path, monkeypatch):
    import server.services.agents.registry as registry_module
    import server.services.execution as execution
    import server.services.triggers as triggers
    import server.utils.timezones as timezones
    from server.services.triggers.service import TriggerService
    from server.services.triggers.store import TriggerStore

    logs = ExecutionAgentLogStore(tmp_path / "execution_agents")
    roster = tmp_path / "execution_agents" / "roster.json"
    roster.write_text(json.dumps(["Email to Alice"]), encoding="utf-8")
    trigger_service = TriggerService(TriggerStore(tmp_path / "triggers.db"))
    for name in ("Email to Alice", "Unknown Agent"):
        trigger_service._store.insert({"agent_name": name, "payload": "p", "status": "active", "created_at": "t", "updated_at": "t"})
    monkeypatch.setattr(registry_module, "_ROSTER_PATH", roster)
    monkeypatch.setattr(execution, "get_execution_agent_logs", lambda: logs)
    monkeypatch.setattr(triggers, "get_trigger_service", lambda: trigger_service)
    monkeypatch.setattr(timezones, "resolve_user_timezone", lambda: timezone.utc)

    d = registry(tmp_path)
    registry_module._initialize(d)
    assert [c.id for c in d.list_cards()] == ["email-to-alice"]
    assert len(trigger_service.list_triggers(agent_id="email-to-alice")) == 1
    # A name the registry doesn't know keeps its old log key.
    assert len(trigger_service.list_triggers(agent_id="unknown-agent")) == 1


def test_update_card_overwrites_only_given_fields(tmp_path):
    from datetime import datetime, timezone
    from server.services.agents.directory import InMemoryDirectory
    from server.services.agents.registry import SqliteAgentDirectory

    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    for directory in (SqliteAgentDirectory(tmp_path / "agents.db"), InMemoryDirectory()):
        card = directory.create("Kwame Book Club Dates", "Ask Kwame which dates work", "ask Kwame", now)
        directory.update_card(card.id, purpose="Book club with Kwame: dates and what to bring", status="dormant")
        got = directory.get(card.id)
        assert got.purpose == "Book club with Kwame: dates and what to bring" and got.status == "dormant"
        assert got.name == "Kwame Book Club Dates" and got.last_status == ""
