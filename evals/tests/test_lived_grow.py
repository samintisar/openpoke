import re

import pytest

from evals.llm import ScriptedLLM, tool_call
from evals.worlds.lived.grow import grow, merge_background, name_background
from evals.worlds.lived.model import Event, Intent, LifePlan, Thread, day_to_datetime

KEYWORDS = ("alpha", "bravo", "charlie", "delta", "echo", "desk", "lamp")
_MUST = re.compile(r"^Must include \(exactly as written\): (.*)$", re.MULTILINE)
_ROSTER = re.compile(r'<agent name="([^"]+)"')
_TURN = re.compile(r"<new_(?:user|agent)_message>\n(.*)\n</new_", re.DOTALL)


def script(messages, sample):
    """Writer: echo the required mentions. IA: per keyword, reuse an agent named after it or create one."""
    prompt = messages[0]["content"]
    if prompt.startswith("Who is sending this:"):
        match = _MUST.search(prompt)
        return {"content": " ".join(re.findall(r'"([^"]+)"', match.group(1))) if match else "hi"}
    if any(m["role"] == "tool" for m in messages):
        return {"content": ""}
    text = _TURN.search(prompt).group(1)
    if "boom" in text:
        raise RuntimeError("model exploded")
    roster = _ROSTER.findall(prompt)
    calls = [tool_call("send_message_to_user", message="on it")]
    for keyword in (k for k in KEYWORDS if k in text.lower()):
        name = next((n for n in roster if keyword in n), f"{keyword} agent")
        calls.append(tool_call("send_message_to_agent", agent_name=name, instructions=f"handle {keyword}"))
    return {"content": "", "tool_calls": calls}


def _llm():
    return ScriptedLLM(script)


def _intent(kind, threads, *mention, message_type="user", forbid=()):
    return Intent(kind=kind, thread_ids=tuple(threads), mention=tuple(mention), forbid=tuple(forbid),
                  message_type=message_type)


THREADS = [Thread(f"t{i}", None, kw, "errand") for i, kw in enumerate(KEYWORDS[:5], 1)]


def _plan(intents, background=()):
    events = [Event(f"e{i:04d}", float(i), intent) for i, intent in enumerate(intents)]
    return LifePlan(seed=1, persona="Sam", contacts=[], threads=THREADS, events=events, background=list(background))


STORY = [
    _intent("result", ["t1"], "alpha", message_type="agent"),  # e0000: no agent yet -> skipped
    _intent("start", ["t1"], "alpha"),  # e0001: creates alpha agent
    _intent("follow_up", ["t1"], "alpha"),  # e0002: reuses it
    _intent("result", ["t1"], "alpha", message_type="agent"),  # e0003
    _intent("multi", ["t2", "t3"], "bravo", "charlie"),  # e0004: two creates, assigned in order
    _intent("start", ["t4"], "boom"),  # e0005: routing error -> skipped
    _intent("start", ["t4"], "delta", forbid=["delta"]),  # e0006: writer fails -> skipped
    _intent("multi", ["t1", "t5"], "alpha", "echo"),  # e0007: one reuse + one create -> counts differ
]


async def test_grow_plays_the_story():
    logs = []
    result = await grow(_plan(STORY), llm=_llm(), model="m", sizes=(1, 3, 4), max_agents=10, log=logs.append)

    names = [a.name for a in result.agents]
    assert names == ["alpha agent", "bravo agent", "charlie agent", "echo agent"]
    ids = {a.name: a.id for a in result.agents}
    assert ids["alpha agent"] == "alpha-agent"
    assert [e for e, _ in result.skipped] == ["e0000", "e0005", "e0006"]
    assert any("e0005" in line for line in logs)

    by_name = {a.name: a for a in result.agents}
    assert by_name["bravo agent"].threads == ["t2"] and by_name["charlie agent"].threads == ["t3"]
    # multi (t1 + t5): alpha already served t1 and keeps only it; the new echo agent takes the uncovered t5
    assert by_name["echo agent"].threads == ["t5"]
    assert by_name["alpha agent"].threads == ["t1"]
    assert by_name["alpha agent"].purpose == "handle alpha"
    assert len(by_name["alpha agent"].history) == 4  # start, follow-up, result, multi

    final = result.snapshots[-1]
    assert final.thread_agents["t1"] == ["alpha-agent"]
    assert final.thread_agents["t5"] == ["echo-agent"]
    assert final.thread_agents["t2"] == ["bravo-agent"]
    assert final.thread_last_day["t1"] == 7.0
    assert final.last_user_threads == ("t1", "t5")


async def test_snapshots_are_taken_at_sizes_and_are_copies():
    result = await grow(_plan(STORY), llm=_llm(), model="m", sizes=(1, 3, 4), max_agents=10, log=lambda _: None)
    assert [(s.size, s.day, len(s.agents)) for s in result.snapshots] == [(1, 1.0, 1), (3, 4.0, 3), (4, 7.0, 4)]

    first, second = result.snapshots[0], result.snapshots[1]
    assert first.agents[0].threads == ["t1"]  # gained t5 later, but the snapshot is a copy
    assert first.agents[0].created_days == 0.0
    assert second.agents[0].created_days == 3.0 and second.agents[0].last_active_days == 1.0
    assert first.thread_agents == {"t1": ["alpha-agent"]}
    assert len(first.transcript) == 2

    result.snapshots[-1].agents[0].history.append(("x", ""))
    assert len(second.agents[0].history) == 3


async def test_a_followed_up_thread_keeps_one_agent():
    plan = _plan([_intent("start", ["t1"], "alpha")] + [_intent("follow_up", ["t1"], "alpha")] * 3)
    result = await grow(plan, llm=_llm(), model="m", sizes=(1,), max_agents=5, log=lambda _: None)
    assert len(result.agents) == 1
    assert result.snapshots[0].thread_agents == {"t1": ["alpha-agent"]}
    assert result.agents[0].last_active_days == 0.0 and result.agents[0].created_days == 3.0


async def test_transcript_holds_user_reply_and_agent_entries():
    result = await grow(_plan(STORY[:4]), llm=_llm(), model="m", sizes=(1,), max_agents=10, log=lambda _: None)
    assert [tag for tag, _, _ in result.transcript] == [
        "user_message", "poke_reply", "user_message", "poke_reply", "agent_message", "poke_reply",
    ]
    tag, text, stamp = result.transcript[4]
    assert text == "[SUCCESS] alpha agent: alpha"
    assert stamp == day_to_datetime(3.0).isoformat()


async def test_growth_stops_at_max_agents():
    result = await grow(_plan(STORY), llm=_llm(), model="m", sizes=(1,), max_agents=2, log=lambda _: None)
    assert len(result.agents) == 3  # the multi turn that crossed the cap created two
    assert result.snapshots[0].size == 1


async def test_events_running_out_raises():
    with pytest.raises(ValueError, match="before sizes"):
        await grow(_plan(STORY), llm=_llm(), model="m", sizes=(1, 50), max_agents=50, log=lambda _: None)


async def test_progress_line_every_25_events():
    logs = []
    plan = _plan([_intent("start", ["t1"], "alpha")] + [_intent("follow_up", ["t1"], "alpha")] * 29)
    await grow(plan, llm=_llm(), model="m", sizes=(1,), max_agents=5, log=logs.append)
    assert logs == ["event 25/30 agents 1 skipped 0"]


BACKGROUND = [
    _intent("start", [], "desk"),
    _intent("start", [], "alpha"),  # names "alpha agent", which the grown roster already has
    _intent("start", [], "nothing here"),  # the IA creates nothing -> dropped
    _intent("start", [], "lamp"),
    _intent("start", [], "desk"),  # same name as the first background agent
]


async def test_name_background_is_deterministic():
    plan = _plan(STORY, BACKGROUND)
    first = await name_background(plan, llm=_llm(), model="m", concurrency=1)
    second = await name_background(plan, llm=_llm(), model="m", concurrency=8)
    assert [(a.id, a.name, a.created_days) for a in first] == [(a.id, a.name, a.created_days) for a in second]
    assert [a.name for a in first] == ["desk agent", "alpha agent", "lamp agent", "desk agent"]
    assert [a.id for a in first] == ["desk-agent", "alpha-agent", "lamp-agent", "desk-agent-2"]
    assert all(a.threads == [] and a.purpose.startswith("handle") for a in first)
    assert all(0 <= a.created_days <= 180 and a.last_active_days == a.created_days for a in first)


async def test_merge_drops_equal_names_and_respects_target():
    plan = _plan(STORY, BACKGROUND)
    grown = await grow(plan, llm=_llm(), model="m", sizes=(4,), max_agents=4, log=lambda _: None)
    background = await name_background(plan, llm=_llm(), model="m")
    snapshot = grown.snapshots[0]

    merged = merge_background(snapshot, background, target=100)
    names = [a.name for a in merged.agents]
    assert sorted(names) == sorted([a.name for a in snapshot.agents] + ["desk agent", "lamp agent"])
    assert len({a.id for a in merged.agents}) == len(merged.agents)
    assert merged.size == 100 and merged.day == snapshot.day
    assert merged.thread_agents == snapshot.thread_agents and merged.transcript == snapshot.transcript
    created = [snapshot.day - a.created_days for a in merged.agents]
    assert created == sorted(created)  # interleaved by created day
    assert all(0 <= a.created_days and 0 <= a.last_active_days for a in merged.agents)

    capped = merge_background(snapshot, background, target=5)
    assert len(capped.agents) == 5
    assert [a.name for a in capped.agents if not a.threads] == ["desk agent"]
    assert len(snapshot.agents) == 4  # the input snapshot is untouched
