"""AgentCard.summary: rendered on the card, searched by retrieval, stored and migrated by the registry."""

import sqlite3
from datetime import datetime, timezone

from server.services.agents.cards import AgentCard
from server.services.agents.directory import InMemoryDirectory
from server.services.agents.registry import SqliteAgentDirectory
from server.services.agents.render import render_card
from server.services.agents.retrieval import score_cards

NOW = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)


def _card(agent_id, name, purpose, summary=""):
    return AgentCard(id=agent_id, name=name, purpose=purpose, summary=summary, created_at=NOW, last_active_at=NOW)


def test_render_shows_the_summary_line():
    text = render_card(_card("t", "Trent - Permit Stuff", "Ask Trent about permits.", "Trent Hollis (contractor): "
                             "permit paperwork; bathroom remodel started, waiting on his quote."), NOW)
    assert "    summary: Trent Hollis (contractor): permit paperwork; bathroom remodel started" in text
    assert "summary:" not in render_card(_card("t", "T", "p"), NOW)


def test_retrieval_reads_the_summary():
    trent = _card("trent", "Trent - Permit Stuff", "Ask Trent about permits.", "Also handles the bathroom remodel.")
    other = _card("ruth", "Ruth Thanksgiving", "Ask Ruth about Thanksgiving.")
    assert score_cards("how's the bathroom remodel going?", [other, trent], NOW)[0][0].id == "trent"


def test_in_memory_update_sets_summary_and_merges_its_people():
    directory = InMemoryDirectory([_card("t", "Trent - Permit Stuff", "Ask Trent about permits.")])
    directory.update_card("t", summary="Trent Hollis is coordinating with Wanda Mills.")
    card = directory.get("t")
    assert card.summary == "Trent Hollis is coordinating with Wanda Mills."
    assert {"hollis", "wanda", "mills"} <= set(card.entities)


def test_sqlite_stores_summary_and_migrates_last_status(tmp_path):
    path = tmp_path / "agents.db"
    with sqlite3.connect(path) as conn:  # the schema before Plan 6c
        conn.execute("CREATE TABLE agents (id TEXT PRIMARY KEY, name TEXT NOT NULL, purpose TEXT NOT NULL, entities TEXT "
                     "NOT NULL DEFAULT '[]', last_status TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'active', "
                     "created_at TEXT NOT NULL, last_active_at TEXT NOT NULL, run_count INTEGER NOT NULL DEFAULT 0)")
        conn.execute("INSERT INTO agents VALUES ('gary', 'Gary Lease', 'Renew the lease', '[]', 'Waiting on Gary', "
                     "'active', '2026-10-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00', 1)")
    directory = SqliteAgentDirectory(path)
    assert directory.get("gary").summary == "Waiting on Gary"
    directory.update_card("gary", summary="Gary Whitman: lease renewal and the parking spot.")
    again = SqliteAgentDirectory(path).get("gary")
    assert again.summary == "Gary Whitman: lease renewal and the parking spot." and "whitman" in again.entities


def test_retrieval_finds_a_card_with_only_last_status():
    old = AgentCard(id="o", name="Old Helper", purpose="Ask about stuff.", last_status="Bathroom remodel quote arrived.",
                    created_at=NOW, last_active_at=NOW)
    other = _card("ruth", "Ruth Thanksgiving", "Ask Ruth about Thanksgiving.")
    assert score_cards("bathroom remodel quote", [other, old], NOW)[0][0].id == "o"


def test_summary_entities_are_replaced_not_accumulated():
    directory = InMemoryDirectory([_card("t", "T", "Ask Trent about permits.")])
    directory.update_card("t", summary="People: Trent Hollis. Decided: Wanda Mills approves. Pending: quote.")
    first = set(directory.get("t").entities)
    assert {"hollis", "wanda", "mills"} <= first and not {"decided", "pending", "people"} & first
    directory.update_card("t", summary="People: Trent Hollis. Topics: Gary Whitman lease.")
    second = set(directory.get("t").entities)
    assert "whitman" in second and not {"wanda", "mills"} & second and "hollis" in second
    for _ in range(5):
        directory.update_card("t", summary="People: Trent Hollis. Topics: Gary Whitman lease.")
    assert set(directory.get("t").entities) == second


def test_sqlite_and_in_memory_agree_on_summary_updates(tmp_path):
    seed = AgentCard(id="t", name="T", purpose="Ask Trent about permits.", entities=["kwame"],
                     created_at=NOW, last_active_at=NOW)
    memory, sqlite_dir = InMemoryDirectory([seed]), SqliteAgentDirectory(tmp_path / "a.db")
    sqlite_dir.add_cards([seed])
    for text in ["  Trent Hollis and Wanda Mills.  ", "Trent Hollis and Gary Whitman.", "Just the quote."]:
        for directory in (memory, sqlite_dir):
            directory.update_card("t", summary=text)
        a, b = memory.get("t"), sqlite_dir.get("t")
        assert (a.summary, a.entities) == (b.summary, b.entities)
    assert memory.get("t").entities == ["kwame"]


_OLD_SCHEMA = ("CREATE TABLE agents (id TEXT PRIMARY KEY, name TEXT NOT NULL, purpose TEXT NOT NULL, entities TEXT "
               "NOT NULL DEFAULT '[]', last_status TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'active', "
               "created_at TEXT NOT NULL, last_active_at TEXT NOT NULL, run_count INTEGER NOT NULL DEFAULT 0)")


def test_migration_is_idempotent_and_handles_empty_and_concurrent_opens(tmp_path):
    import threading

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute(_OLD_SCHEMA)
    SqliteAgentDirectory(path)  # empty agents table
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO agents (id, name, purpose, summary, created_at, last_active_at) VALUES "
                     "('g', 'G', 'p', 'kept', '2026-10-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00')")
    SqliteAgentDirectory(path)  # already migrated: must not touch summaries
    assert SqliteAgentDirectory(path).get("g").summary == "kept"

    race = tmp_path / "race.db"
    with sqlite3.connect(race) as conn:
        conn.execute(_OLD_SCHEMA)
        conn.execute("INSERT INTO agents (id, name, purpose, last_status, created_at, last_active_at) VALUES "
                     "('x', 'X', 'p', 'old status', '2026-10-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00')")
    errors, barrier = [], threading.Barrier(2)

    def open_it():
        barrier.wait()
        try:
            SqliteAgentDirectory(race)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=open_it) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and SqliteAgentDirectory(race).get("x").summary == "old status"
