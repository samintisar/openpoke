"""Retrieval: BM25, fuzzy entity matching, recency, boosts and selection caps."""

from datetime import datetime, timedelta, timezone

import pytest

from server.services.agents import (
    AgentCard,
    RetrievalConfig,
    build_query,
    pinned_from_agent_message,
    score_cards,
    search,
    select,
    tokenize,
)
from server.services.agents.retrieval import edit_distance_le1

NOW = datetime(2026, 6, 26, 12, 0, tzinfo=timezone.utc)


def card(agent_id, name, purpose, entities, days=10.0, status="active"):
    when = NOW - timedelta(days=days)
    return AgentCard(agent_id, name, purpose, entities, status=status, created_at=when, last_active_at=when)


CARDS = [
    card("kwame-book-club-dates", "Kwame Book Club Dates", "Ask Kwame which dates work for book club", ["kwame"]),
    card("kwame-podcast", "Kwame Podcast Guest Spot Reply Draft", "Reply to Kwame about the podcast guest spot", ["kwame"]),
    card("beatriz-trivia", "Beatriz Trivia Night Note", "Draft an email to Beatriz about trivia night", ["beatriz"]),
    card("beatriz-big-sur", "Beatriz Big Sur Trip Push Request", "Ask Beatriz to push the Big Sur trip", ["beatriz", "big", "sur"]),
    card("gary-lease", "Gary Whitman Lease Renewal Call", "Call Gary Whitman about the lease renewal", ["gary", "whitman"]),
    card("maya-okrs", "Maya Quarterly OKRs Draft", "Ask Maya for the quarterly OKRs draft", ["maya", "okrs"]),
]


def names(cards):
    return [c.id for c in cards]


def test_tokenize():
    assert tokenize("Gary's OKRs for the Book-Club, gary@whitman.com") == ["gary", "okr", "book", "club", "gary@whitman.com"]


def test_edit_distance_le1():
    assert edit_distance_le1("baetriz", "beatriz")  # transposition
    assert edit_distance_le1("kwam", "kwame")
    assert edit_distance_le1("maya", "mayo")
    assert not edit_distance_le1("maya", "nora")
    assert not edit_distance_le1("beatriz", "bea")


def test_bm25_orders_by_topic():
    scored = score_cards("when is book club with kwame", CARDS, NOW)
    assert names(c for c, _ in scored[:2]) == ["kwame-book-club-dates", "kwame-podcast"]
    assert scored[0][1] > scored[1][1] > scored[-1][1]


def test_fuzzy_typo_finds_beatriz():
    top = select("any update from baetriz on trivia night?", CARDS, NOW)
    assert top[0].id == "beatriz-trivia"


def test_fuzzy_needs_min_length_and_counts_half():
    cfg = RetrievalConfig(w_kw=0.0, w_rec=0.0)
    exact = dict((c.id, s) for c, s in score_cards("maya", CARDS, NOW, cfg=cfg))
    fuzzy = dict((c.id, s) for c, s in score_cards("mayo", CARDS, NOW, cfg=cfg))
    assert exact["maya-okrs"] == fuzzy["maya-okrs"] == 0.40  # normalized by the max in the set
    short = [card("ben", "Ben Okafor Camping Weekend", "Camping with Ben", ["ben", "okafor"])]
    assert score_cards("bem", short, NOW, cfg=cfg)[0][1] == 0.0


def test_recency_and_last_routed_boost():
    twins = [
        card("old", "Ben Okafor Ski Trip Reply", "Reply to Ben about the ski trip", ["ben"], days=30),
        card("new", "Ben Okafor Ski Trip Reply", "Reply to Ben about the ski trip", ["ben"], days=1),
    ]
    scored = score_cards("ben ski trip", twins, NOW)
    assert names(c for c, _ in scored) == ["new", "old"]
    boosted = score_cards("ben ski trip", twins, NOW, last_routed=["old"])
    assert names(c for c, _ in boosted) == ["old", "new"]
    assert boosted[0][1] - scored[1][1] == pytest.approx(RetrievalConfig().last_routed_boost)


def test_select_pinned_first_then_hot_set_capped():
    hot = [card(f"hot-{i}", f"Errand {i}", "Price check", [], days=i / (24 * 60)) for i in range(7)]
    cards = CARDS + hot
    chosen = select("gary lease", cards, NOW, pinned=["maya-okrs"])
    assert chosen[0].id == "maya-okrs"
    assert names(chosen[1:6]) == ["hot-0", "hot-1", "hot-2", "hot-3", "hot-4"]
    assert "hot-5" not in names(chosen[:6])
    assert chosen[6].id == "gary-lease"
    assert len(names(chosen)) == len(set(names(chosen)))


def test_select_open_work_joins_hot_set():
    chosen = select("unrelated", CARDS, NOW, open_work=["kwame-podcast"])
    assert chosen[0].id == "kwame-podcast"


def test_select_top_k_cap():
    many = [card(f"ben-{i}", f"Ben Okafor Topic {i}", "Ask Ben", ["ben"]) for i in range(20)]
    assert len(select("ben", many, NOW, cfg=RetrievalConfig(k=8))) == 8


def test_select_min_score_and_archived_exclusion():
    cards = CARDS + [
        card("gary-archived", "Gary Whitman Deposit", "Get the lease deposit back from Gary Whitman", ["gary", "whitman"], status="archived"),
    ]
    chosen = names(select("gary whitman lease", cards, NOW))
    assert chosen[0] == "gary-lease"
    assert "gary-archived" not in chosen
    assert "kwame-podcast" not in chosen  # irrelevant and 10 days old: below min_score
    assert "gary-archived" in names(search("gary whitman lease", cards, NOW))


def test_search_limit():
    many = [card(f"ben-{i}", f"Ben Okafor Topic {i}", "Ask Ben", ["ben"]) for i in range(20)]
    assert len(search("ben", many, NOW)) == 10
    assert len(search("ben", many, NOW, limit=3)) == 3


def test_build_query_uses_last_three_turns():
    query = build_query("did she reply?", ["one", "two", "three", "four"])
    assert query.splitlines() == ["did she reply?", "two", "three", "four"]


def test_pinned_from_agent_message():
    message = (
        "[SUCCESS] Beatriz Trivia Night Note: Draft ready.\n"
        "[FAILED] gary whitman lease renewal call: Could not find the thread.\n"
        "[SUCCESS] Unknown Agent: done"
    )
    assert names(pinned_from_agent_message(message, CARDS)) == ["beatriz-trivia", "gary-lease"]
