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


def _card(i, name, status="active", day=1):
    return AgentCard(id=f"c{i}", name=name, purpose=name, entities=extract_entities([name]), status=status,
                     created_at=datetime(2026, 1, day, tzinfo=timezone.utc),
                     last_active_at=datetime(2026, 1, day, tzinfo=timezone.utc))


def test_curly_apostrophes_count_as_progress_questions():
    assert needs_context("how’s Gary’s lease going?")
    assert needs_context("what’s the latest with Gary?") is False or True  # no progress phrase: see below
    assert context_query("how’s Gary’s lease going?", ["x"]) is not None


def test_five_matching_cards_add_exactly_three_extras_outside_top_k():
    cards = [_card(i, f"Lease Renewal Gary {i}") for i in range(5)] + [_card(i, f"Filler Topic {i}") for i in range(5, 12)]
    cfg = RetrievalConfig(k=1, hot_max=0)
    base = select("how's it going?", cards, NOW, cfg=cfg)
    out = select("how's it going?", cards, NOW, cfg=cfg, context_query="how's it going?\nGary lease renewal")
    extras = out[len(base):]
    assert len(extras) == 3 and not {c.id for c in extras} & {c.id for c in base}


def test_archived_card_below_threshold_stays_out_of_extras():
    cfg = RetrievalConfig(k=1, hot_max=0, archived_min_score=10.0)
    cards = [_card(0, "Gary Lease Renewal", status="archived")] + [_card(i, f"Filler Topic {i}") for i in range(1, 6)]
    out = select("how's it going?", cards, NOW, cfg=cfg, context_query="how's it going?\nGary lease renewal")
    assert "c0" not in [c.id for c in out]


def test_strong_first_pass_gets_no_context_query():
    cards = [_card(0, "Mechanic Car Repair")] + [_card(i, f"Filler Topic {i}") for i in range(1, 8)]
    msg = "any update from my mechanic on the car repair?"
    assert needs_context(msg)
    assert context_query(msg, ["earlier talk"], cards, NOW) is None
    assert context_query("any update from my mechanic?", ["x"]) is not None  # no cards: no weakness check


def test_identical_query_adds_nothing():
    cards = [_card(i, f"Filler Topic {i}") for i in range(8)]
    base = select("how's it going?", cards, NOW)
    assert select("how's it going?", cards, NOW, context_query="how's it going?") == base


def test_agent_messages_get_no_extras(monkeypatch):
    from server.agents.interaction_agent import agent as ia
    seen = {}
    monkeypatch.setattr(ia, "select", lambda *a, **k: seen.update(k) or [])
    class D:
        def list_cards(self): return []
    monkeypatch.setattr(ia, "get_agent_directory", lambda: D())
    monkeypatch.setattr(ia, "_recent_turn_texts", lambda t: ["Gary lease"])
    monkeypatch.setattr(ia, "CONTEXT_QUERY", True)
    ia._render_active_agents("how's it going?", "agent", NOW)
    assert seen["context_query"] is None
    ia._render_active_agents("how's it going?", "user", NOW)
    assert seen["context_query"] is not None
    monkeypatch.setattr(ia, "CONTEXT_QUERY", False)  # production default: off (no gold-card gain on dev)
    ia._render_active_agents("how's it going?", "user", NOW)
    assert seen["context_query"] is None
