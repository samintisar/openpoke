"""Agent cards: entity extraction, name normalization, slug ids and rendering."""

from datetime import datetime, timedelta, timezone

from server.services.agents import AgentCard, extract_entities, normalize_name, render_cards, slug_id

NOW = datetime(2026, 6, 26, 12, 0, tzinfo=timezone.utc)


def test_extract_entities_takes_names_not_sentence_openers():
    assert extract_entities(["Draft a reply to Gary Whitman about the lease"]) == ["gary", "whitman"]


def test_extract_entities_emails_dedupe_and_stoplist():
    texts = [
        "Email gary.whitman@example.com about the lease renewal on Monday in March.",
        "Please ask Gary if I can call him Friday. Can Gary's landlord Priya Shah join?",
    ]
    assert extract_entities(texts) == ["gary.whitman@example.com", "gary", "priya", "shah"]


def test_extract_entities_skips_title_case_labels_and_lowercase_openers():
    assert extract_entities(["Kwame Book Club Dates"]) == []
    assert extract_entities(["Buy the cheapest one. Compare prices on Amazon and buy it."]) == ["amazon"]
    assert extract_entities(["Beatriz is hosting trivia night."]) == ["beatriz"]


def test_normalize_name():
    assert normalize_name("  Maya's  Quarterly-OKRs, Draft! ") == "mayas quarterly okrs draft"


def test_slug_id_collisions():
    assert slug_id("Gary Whitman Lease Renewal Call", []) == "gary-whitman-lease-renewal-call"
    taken = {"kwame-book-club-dates", "kwame-book-club-dates-2"}
    assert slug_id("Kwame Book Club Dates", taken) == "kwame-book-club-dates-3"
    assert slug_id("!!!", []) == "agent"


def test_render_cards_format():
    cards = [
        AgentCard(
            "beatriz-trivia-night-note",
            "Beatriz Trivia Night Note",
            "Draft an email to Beatriz about trivia night " + "x" * 300,
            ["beatriz"],
            last_status="Draft ready for review",
            last_active_at=NOW - timedelta(hours=2),
        ),
        AgentCard("kwame-book-club-dates", 'Kwame "Book" Club', "Find dates", last_active_at=NOW - timedelta(days=3)),
    ]
    text = render_cards(cards, 240, NOW)
    lines = text.splitlines()
    assert lines[0] == '<active_agents shown="2" total="240">'
    assert lines[1] == '  <agent id="beatriz-trivia-night-note" name="Beatriz Trivia Night Note" last_active="2h ago">'
    assert lines[2].startswith("    purpose: Draft an email to Beatriz") and lines[2].endswith("…")
    assert len(lines[2].removeprefix("    purpose: ")) == 160
    assert lines[3] == "    status: Draft ready for review"
    assert lines[4] == "  </agent>"
    assert 'name="Kwame &quot;Book&quot; Club" last_active="3d ago"' in lines[5]
    assert "status:" not in "\n".join(lines[5:])
    assert lines[-1] == "</active_agents>"


def test_render_cards_empty():
    assert render_cards([], 12, NOW) == '<active_agents shown="0" total="12">None</active_agents>'
