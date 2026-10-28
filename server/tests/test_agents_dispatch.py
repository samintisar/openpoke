"""Dispatch semantics, the in-memory directory and the tool schemas."""

from datetime import datetime, timedelta, timezone

from server.services.agents import AgentCard, InMemoryDirectory, TOOL_SCHEMAS, dispatch, search_payload

NOW = datetime(2026, 6, 26, 12, 0, tzinfo=timezone.utc)
EARLIER = NOW - timedelta(days=4)


def directory():
    return InMemoryDirectory(
        [
            AgentCard(
                "kwame-book-club-dates",
                "Kwame Book Club Dates",
                "Draft an email to Kwame asking what dates work for book club",
                ["kwame"],
                created_at=EARLIER,
                last_active_at=EARLIER,
                run_count=2,
            ),
            AgentCard(
                "beatriz-trivia-night-note",
                "Beatriz Trivia Night Note",
                "Draft an email to Beatriz about trivia night",
                ["beatriz"],
                status="dormant",
                created_at=EARLIER,
                last_active_at=EARLIER,
                run_count=1,
            ),
        ]
    )


def test_reuse_by_id_touches_card():
    d = directory()
    result = dispatch(d, instructions="Also loop in Sofia Reyes", agent_id="kwame-book-club-dates", now=NOW)
    assert result.action == "reused"
    assert result.payload == {
        "status": "submitted",
        "agent_id": "kwame-book-club-dates",
        "agent_name": "Kwame Book Club Dates",
        "new_agent_created": False,
    }
    card = d.get("kwame-book-club-dates")
    assert card.last_active_at == NOW and card.run_count == 3
    assert card.entities == ["kwame", "sofia", "reyes"]


def test_reuse_revives_dormant_card():
    d = directory()
    dispatch(d, instructions="Any reply?", agent_id="beatriz-trivia-night-note", now=NOW)
    assert d.get("beatriz-trivia-night-note").status == "active"


def test_unknown_id_lists_nearest():
    d = directory()
    result = dispatch(d, instructions="Ping her", agent_id="beatriz-trivia", now=NOW)
    assert result.action == "error" and result.card is None
    assert "beatriz-trivia" in result.payload["error"]
    assert 'id="beatriz-trivia-night-note"' in result.payload["nearest_agents"]
    assert len(d.list_cards()) == 2


def test_create_requires_name_and_purpose():
    d = directory()
    for kwargs in ({}, {"name": "Gary Whitman Lease Renewal Call"}, {"purpose": "Call Gary", "name": "  "}):
        result = dispatch(d, instructions="Call Gary", now=NOW, **kwargs)
        assert result.action == "error"
        assert "name and purpose" in result.payload["error"]
    assert len(d.list_cards()) == 2


def test_possible_duplicate_then_force_new():
    d = directory()
    kwargs = dict(
        instructions="Ask Kwame if they need me to bring anything to book club",
        name="Kwame Book Club Bring Item Note",
        purpose="Ask Kwame if they need me to bring anything to book club",
        now=NOW,
    )
    result = dispatch(d, **kwargs)
    assert result.action == "possible_duplicate"
    assert result.card.id == "kwame-book-club-dates"
    assert 'id="kwame-book-club-dates"' in result.payload["candidate"]
    assert 'agent_id="kwame-book-club-dates"' in result.payload["instructions"]
    assert "force_new=true" in result.payload["instructions"]
    assert len(d.list_cards()) == 2

    forced = dispatch(d, force_new=True, **kwargs)
    assert forced.action == "created"
    assert forced.card.id == "kwame-book-club-bring-item-note"


def test_create_extracts_entities_and_collides_ids():
    d = directory()
    result = dispatch(
        d,
        instructions="Find the thread with gary.whitman@example.com and draft a reply.",
        name="Gary Whitman Lease Renewal Call",
        purpose="Schedule a call with Gary Whitman about renewing the lease",
        now=NOW,
    )
    assert result.action == "created"
    assert result.payload["new_agent_created"] is True
    card = result.card
    assert card.id == "gary-whitman-lease-renewal-call"
    assert card.entities == ["gary.whitman@example.com", "gary", "whitman"]
    assert card.created_at == card.last_active_at == NOW and card.run_count == 1

    again = dispatch(
        d, instructions="x", name="Gary Whitman Lease Renewal Call", purpose="Same name", force_new=True, now=NOW
    )
    assert again.card.id == "gary-whitman-lease-renewal-call-2"


def test_in_memory_directory_copies_cards_and_tracks_last_routed():
    original = AgentCard("a", "Maya Quarterly OKRs Draft", "OKRs", ["maya"], last_active_at=EARLIER)
    d = InMemoryDirectory([original])
    d.touch("a", "Loop in Nora", NOW)
    assert original.entities == ["maya"] and original.last_active_at == EARLIER
    assert d.last_routed() == []
    d.set_last_routed(["a"])
    assert d.last_routed() == ["a"]


def test_search_payload_renders_results():
    text = search_payload(directory(), "baetriz trivia", NOW)
    assert text.startswith('<active_agents shown="')
    assert 'total="2"' in text and 'id="beatriz-trivia-night-note"' in text


def test_schema_shape():
    by_name = {schema["function"]["name"]: schema["function"] for schema in TOOL_SCHEMAS}
    assert list(by_name) == [
        "send_message_to_agent", "search_agents", "send_message_to_user", "send_draft", "approve_send", "wait"
    ]
    agent = by_name["send_message_to_agent"]["parameters"]
    assert agent["required"] == ["instructions"]
    assert set(agent["properties"]) == {"instructions", "agent_id", "name", "purpose", "force_new"}
    assert agent["properties"]["force_new"]["type"] == "boolean"
    assert agent["additionalProperties"] is False
    description = by_name["send_message_to_agent"]["description"]
    assert "agent_id" in description and "name" in description and "purpose" in description
    assert by_name["search_agents"]["parameters"]["required"] == ["query"]
    draft = by_name["send_draft"]["parameters"]
    assert "required" not in draft  # either action_id, or to/subject/body (checked by the tool)
    assert set(draft["properties"]) == {"action_id", "to", "subject", "body"}
    assert by_name["approve_send"]["parameters"]["required"] == ["action_id"]
    assert by_name["wait"]["parameters"]["required"] == ["reason"]
    assert by_name["send_message_to_user"]["parameters"]["required"] == ["message"]


def test_force_new_string_false_does_not_bypass_the_guard():
    from server.services.agents.cards import AgentCard
    from server.services.agents.directory import InMemoryDirectory
    from server.services.agents.dispatch import dispatch
    from datetime import datetime, timezone

    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    card = AgentCard(id="kwame-book-club-dates", name="Kwame Book Club Dates", purpose="Ask Kwame which dates work for book club",
                     entities=["kwame"], created_at=now, last_active_at=now)
    directory = InMemoryDirectory([card])
    kwargs = dict(instructions="ask Kwame what to bring", name="Kwame Book Club Bring Item Note",
                  purpose="Ask Kwame what to bring to book club", now=now)
    assert dispatch(directory, force_new="false", **kwargs).action == "possible_duplicate"
    assert dispatch(directory, force_new="True", **kwargs).action == "created"
