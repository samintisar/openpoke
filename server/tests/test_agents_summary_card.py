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
