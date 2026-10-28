"""Create-time duplicate guard."""

from datetime import datetime, timedelta, timezone

from server.services.agents import AgentCard, find_duplicate

NOW = datetime(2026, 6, 26, 12, 0, tzinfo=timezone.utc)


def card(agent_id, name, purpose, entities, status="active"):
    when = NOW - timedelta(days=5)
    return AgentCard(agent_id, name, purpose, entities, status=status, created_at=when, last_active_at=when)


CARDS = [
    card("kwame-book-club-dates", "Kwame Book Club Dates", "Draft an email to Kwame asking what dates work for book club", ["kwame"]),
    card("maya-okrs", "Maya Quarterly OKRs Draft", "Send an email to Maya asking for the latest draft of the quarterly OKRs", ["maya", "okrs"]),
    card("elena-gifts", "Elena Christmas Gifts", "Help sort out Christmas gifts for Elena", ["christmas", "elena"]),
    card("marco-book-club", "Marco Book Club Note", "Tell Marco the book club moved to Thursday", ["marco"]),
    card("gary-lease", "Gary Whitman Lease Renewal Call", "Call Gary Whitman about the lease renewal", ["gary", "whitman"], status="archived"),
]


def test_near_identical_name_is_duplicate():
    dup = find_duplicate("Kwame Book Club Date", "Check with him on timing", CARDS, NOW)
    assert dup is not None and dup.id == "kwame-book-club-dates"


def test_archived_cards_are_included():
    dup = find_duplicate("Gary Whitman Lease Renewal Call!", "Call about the lease", CARDS, NOW)
    assert dup is not None and dup.id == "gary-lease"


def test_same_person_same_topic_is_duplicate():
    dup = find_duplicate(
        "Kwame Book Club Bring Item Note", "Ask Kwame if they need me to bring anything to book club", CARDS, NOW
    )
    assert dup is not None and dup.id == "kwame-book-club-dates"
    dup = find_duplicate("Maya Nora OKR Signoff Note", "Tell Maya that Nora signed off on the OKRs", CARDS, NOW)
    assert dup is not None and dup.id == "maya-okrs"


def test_same_person_new_topic_is_not_duplicate():
    assert find_duplicate("Elena Moving Help", "Tell Elena I'm excited about the moving help", CARDS, NOW) is None


def test_same_topic_other_person_is_not_duplicate():
    assert find_duplicate("Anya Book Club Snacks", "Ask Anya what snacks to bring to book club", CARDS, NOW) is None


def test_no_cards():
    assert find_duplicate("Kwame Book Club Dates", "Ask Kwame about dates", [], NOW) is None
