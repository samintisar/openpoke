from datetime import datetime, timezone

from server.services.agents.cards import AgentCard, extract_entities
from server.services.agents.retrieval import RetrievalConfig, context_query, needs_context, select

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


def test_needs_context():
    assert needs_context("how's the parking going?")
    assert needs_context("any word back on the biweekly?")
    assert needs_context("book the flight if it's under budget")  # names no one
    assert not needs_context("ask Kwame which dates work for book club")


def test_context_query_adds_the_last_three_turns_only_when_needed():
    recent = ["one", "two", "Gary said the parking spot is still open", "ok"]
    assert context_query("how's it going?", recent) == "how's it going?\ntwo\nGary said the parking spot is still open\nok"
    assert context_query("ask Kwame about book club", recent) is None


def test_select_merges_up_to_three_context_cards():
    def card(i, name):
        return AgentCard(id=f"c{i}", name=name, purpose=name, entities=extract_entities([name]),
                         created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                         last_active_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    cards = [card(0, "Gary Whitman Lease Renewal Call")] + [card(i, f"Filler Topic {i}") for i in range(1, 12)]
    cfg = RetrievalConfig(k=2)
    without = select("how's it going?", cards, NOW, cfg=cfg)
    with_context = select("how's it going?", cards, NOW, cfg=cfg, context_query="how's it going?\nGary lease renewal")
    assert "c0" not in [c.id for c in without] and "c0" in [c.id for c in with_context]
    assert len(with_context) <= len(without) + 3
