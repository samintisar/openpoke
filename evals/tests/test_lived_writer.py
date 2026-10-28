from evals.llm import ScriptedLLM
from evals.worlds.lived.model import Intent
from evals.worlds.lived.writer import AGENT_SYSTEM, USER_SYSTEM, write

PERSONA = "Sam, product manager in Toronto"
INTENT = Intent(
    kind="follow_up", thread_ids=("t001",), contacts=("Alice Chen",), topic="contract renewal",
    detail="say you can sign on Friday", mention=("Alice",), forbid=("Chen",),
)


class Recorder(ScriptedLLM):
    """ScriptedLLM that also records each call's system prompt, user prompt and sample."""

    def __init__(self, replies):
        self.calls = []
        super().__init__(lambda messages, sample: {"content": replies[min(len(self.calls) - 1, len(replies) - 1)]})

    async def complete(self, *, model, system, messages, tools, sample):
        self.calls.append({"system": system, "user": messages[0]["content"], "tools": tools, "sample": sample})
        return await super().complete(model=model, system=system, messages=messages, tools=tools, sample=sample)


async def _write(llm, intent=INTENT, tail=(), sample=0):
    return await write(intent, transcript_tail=list(tail), persona=PERSONA, llm=llm, model="m", sample=sample)


async def test_valid_message_is_returned_with_quotes_stripped():
    llm = Recorder(['  "alice can sign friday, tell her pls"  '])
    assert await _write(llm) == "alice can sign friday, tell her pls"
    assert [c["sample"] for c in llm.calls] == [0]
    call = llm.calls[0]
    assert call["system"] == USER_SYSTEM and call["tools"] == []
    assert PERSONA in call["user"]
    assert 'Must include (exactly as written): "Alice"' in call["user"]
    assert 'Must not include: "Chen"' in call["user"]
    assert "contract renewal" in call["user"] and "sign on Friday" in call["user"]


async def test_forbidden_mention_retries_with_the_next_sample():
    llm = Recorder(["tell alice chen friday works", "tell alice friday works"])
    assert await _write(llm, sample=4) == "tell alice friday works"
    assert [c["sample"] for c in llm.calls] == [4, 5]


async def test_three_failures_return_none():
    llm = Recorder(["no name here", "", "still CHEN"])
    assert await _write(llm) is None
    assert [c["sample"] for c in llm.calls] == [0, 1, 2]


async def test_agent_intent_uses_the_report_prompt():
    intent = Intent(
        kind="result", thread_ids=("t001",), contacts=("Alice Chen",), topic="contract renewal",
        detail="she can do Thursday at 3", mention=("Alice",), message_type="agent",
    )
    llm = Recorder(["Alice replied: she can do Thursday at 3"])
    assert await _write(llm, intent=intent) == "Alice replied: she can do Thursday at 3"
    assert llm.calls[0]["system"] == AGENT_SYSTEM
    assert "Report the outcome of the task" in llm.calls[0]["user"]


async def test_transcript_tail_renders_only_the_last_six_entries():
    tail = [("user_message", f"msg {i}", "2026-01-01T00:00:00") for i in range(8)]
    tail[-1] = ("poke_reply", "msg 7")  # 2- and 3-tuples both work
    llm = Recorder(["alice ok"])
    await _write(llm, tail=tail)
    user = llm.calls[0]["user"]
    assert "msg 0" not in user and "msg 1" not in user
    assert "user_message: msg 2" in user and "poke_reply: msg 7" in user
    assert "2026-01-01" not in user


async def test_probe_kinds_fall_back_to_the_detail():
    intent = Intent(kind="vague_reference", thread_ids=("t001",), detail="ask how that thing is going, no names")
    llm = Recorder(["hows that thing going"])
    assert await _write(llm, intent=intent) == "hows that thing going"
    assert "ask how that thing is going, no names" in llm.calls[0]["user"]
