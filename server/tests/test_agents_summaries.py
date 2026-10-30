import json
from datetime import datetime, timezone

from server.services.agents import summaries
from server.services.agents.cards import AgentCard
from server.services.agents.directory import InMemoryDirectory
from server.services.agents.summaries import (
    SUMMARY_PROMPT, SUMMARY_WORDS, SummaryError, build_summary_input, parse_summary, refresh_card_summary,
)

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


def _dir():
    return InMemoryDirectory([AgentCard(id="trent", name="Trent - Permit Stuff", purpose="Ask Trent about permits.",
                                        created_at=NOW, last_active_at=NOW)])


def test_prompt_and_input_shape():
    assert '{"summary": "..."}' in SUMMARY_PROMPT and "<untrusted_email>" in SUMMARY_PROMPT
    text = build_summary_input("Trent - Permit Stuff", "Ask Trent about permits.", "", "Ask about the remodel", "")
    assert "<current_summary>\n(none)\n</current_summary>" in text and "(no result yet)" in text


def test_parse_clips_to_the_word_limit_and_rejects_junk():
    long = " ".join(["word"] * (SUMMARY_WORDS + 30))
    assert len(parse_summary(json.dumps({"summary": long})).split()) == SUMMARY_WORDS
    assert parse_summary('```json\n{"summary": "Trent:  remodel\\npending"}\n```') == "Trent: remodel pending"
    for bad in ("", "no json", '{"summary": ""}', '["x"]'):
        try:
            parse_summary(bad)
        except SummaryError:
            continue
        raise AssertionError(bad)


async def test_refresh_writes_the_summary():
    directory = _dir()
    seen = []

    async def complete(system, user):
        seen.append(user)
        return json.dumps({"summary": "Trent Hollis: permits; bathroom remodel quote pending."})

    assert await refresh_card_summary("trent", "Ask Trent about the bathroom remodel", "[SUCCESS] Trent - Permit "
                                      "Stuff: He'll send a quote Friday.", directory=directory, complete=complete)
    assert directory.get("trent").summary == "Trent Hollis: permits; bathroom remodel quote pending."
    assert "He'll send a quote Friday." in seen[0]


async def test_failed_update_keeps_the_previous_summary_and_never_raises():
    directory = _dir()
    directory.update_card("trent", summary="Old summary.")

    async def broken(system, user):
        raise RuntimeError("down")

    assert not await refresh_card_summary("trent", "x", "y", directory=directory, complete=broken)
    assert directory.get("trent").summary == "Old summary."
    assert not await refresh_card_summary("missing", "x", "y", directory=directory, complete=broken)


def test_runner_schedules_a_summary_update(monkeypatch):
    from server.services.execution import runner as runner_module
    assert not hasattr(runner_module, "_record_status")
    assert runner_module.refresh_card_summary is summaries.refresh_card_summary
