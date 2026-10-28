import json
import re
from datetime import datetime

import pytest

from evals.grading import grade
from evals.layers.retrieval import run_retrieval_case
from evals.layers.routing import MAX_ITERATIONS, build_messages, run_routing_case
from evals.llm import ScriptedLLM, tool_call
from evals.systems.baseline import BaselineSystem
from evals.worlds.generator import build_world, generate_cases


def setup(kind="follow_up"):
    full = build_world(1)
    world = full.truncated(20)
    case = next(c for c in generate_cases(full, per_kind=2) if c.kind == kind)
    return world, case


def tool_turns(messages):
    return sum(1 for m in messages if m["role"] == "tool")


async def test_dispatch_after_acknowledgement_is_correct():
    world, case = setup()
    target = world.by_id()[case.gold_targets[0]].name

    def script(messages, sample):
        seen = tool_turns(messages)
        if seen == 0:
            return {"content": "", "tool_calls": [tool_call("send_message_to_user", message="on it")]}
        if seen == 1:
            return {"content": "", "tool_calls": [tool_call("send_message_to_agent", agent_name=target, instructions="do it")]}
        return {"content": ""}

    decision = await run_routing_case(case, world, BaselineSystem(), ScriptedLLM(script), model="m", sample=0)
    assert decision.dispatched_ids == [case.gold_targets[0]]
    assert decision.created_names == []
    assert decision.user_messages == ["on it"]
    assert decision.iterations == 3
    assert grade(case, decision) == "correct"


async def test_unknown_name_creates_and_stub_says_so():
    world, case = setup()
    captured = []

    def script(messages, sample):
        captured.append(list(messages))
        if tool_turns(messages) == 0:
            return {"content": "", "tool_calls": [tool_call("send_message_to_agent", agent_name="Totally new", instructions="x")]}
        return {"content": ""}

    decision = await run_routing_case(case, world, BaselineSystem(), ScriptedLLM(script), model="m", sample=0)
    assert decision.created_names == ["Totally new"]
    assert grade(case, decision) == "duplicate_create"
    stub = json.loads(captured[1][-1]["content"])
    assert stub["result"]["new_agent_created"] is True


async def test_question_without_dispatch_is_a_clarification():
    world, case = setup("same_name_people")

    def script(messages, sample):
        if tool_turns(messages) == 0:
            return {"content": "", "tool_calls": [tool_call("send_message_to_user", message="which one do you mean?")]}
        return {"content": ""}

    decision = await run_routing_case(case, world, BaselineSystem(), ScriptedLLM(script), model="m", sample=0)
    assert decision.clarified is True


async def test_iteration_cap():
    world, case = setup()
    llm = ScriptedLLM(lambda messages, sample: {"content": "", "tool_calls": [tool_call("wait", reason="x")]})
    decision = await run_routing_case(case, world, BaselineSystem(), llm, model="m", sample=0)
    assert decision.iterations == MAX_ITERATIONS


def test_prompt_sections_follow_production_order():
    world, case = setup("vague_reference")
    content = build_messages(case, world, BaselineSystem())[0]["content"]
    assert content.index("<conversation_history>") < content.index("<active_agents>") < content.index("<new_user_message>")
    assert case.message in content


def test_agent_results_use_the_agent_tag():
    world, case = setup("agent_result")
    assert "<new_agent_message>" in build_messages(case, world, BaselineSystem())[0]["content"]


def test_retrieval_baseline_shows_everything():
    world, case = setup()
    row = run_retrieval_case(case, world, BaselineSystem())
    assert row["shown_count"] == 20
    assert row["recall"] == 1.0
    assert row["roster_tokens"] > 0


def test_retrieval_create_cases_have_no_recall():
    world, case = setup("new_task")
    assert run_retrieval_case(case, world, BaselineSystem())["recall"] is None


async def test_gold_name_with_trailing_space_creates_like_production():
    world, case = setup()
    target = world.by_id()[case.gold_targets[0]].name + " "

    def script(messages, sample):
        if tool_turns(messages) == 0:
            return {"content": "", "tool_calls": [tool_call("send_message_to_agent", agent_name=target, instructions="x")]}
        return {"content": ""}

    decision = await run_routing_case(case, world, BaselineSystem(), ScriptedLLM(script), model="m", sample=0)
    assert decision.created_names == [target]
    assert decision.dispatched_ids == []


def raw_call(name, arguments):
    return {"id": f"raw_{name}", "type": "function", "function": {"name": name, "arguments": arguments}}


def first_turn_only(*calls, content=""):
    def script(messages, sample):
        if tool_turns(messages) == 0:
            return {"content": "", "tool_calls": list(calls)}
        return {"content": content}

    return script


async def run(case, world, script):
    captured = []

    def recording(messages, sample):
        captured.append(list(messages))
        return script(messages, sample)

    decision = await run_routing_case(case, world, BaselineSystem(), ScriptedLLM(recording), model="m", sample=0)
    return decision, captured


@pytest.mark.parametrize(
    "arguments, fragment",
    [
        ('{"agent_name": "Gold", "instru', "invalid json"),
        (json.dumps({"agent_name": "GOLD"}), "instructions"),
        (json.dumps({"agent_name": "GOLD", "instructions": "x", "priority": "high"}), "priority"),
    ],
    ids=["truncated_json", "missing_key", "extra_key"],
)
async def test_invalid_tool_arguments_are_not_actions(arguments, fragment):
    world, case = setup()
    gold_name = world.by_id()[case.gold_targets[0]].name
    arguments = arguments.replace("GOLD", gold_name)
    decision, captured = await run(case, world, first_turn_only(raw_call("send_message_to_agent", arguments)))
    assert decision.created_names == [] and decision.dispatched_ids == []
    tool_message = json.loads(captured[1][-1]["content"])
    assert tool_message["status"] == "error"
    assert fragment in tool_message["error"]["error"]
    assert "result" not in tool_message
    assert fragment in decision.tool_calls[0]["error"]


async def test_invalid_user_message_is_not_shown():
    world, case = setup("same_name_people")
    call = raw_call("send_message_to_user", json.dumps({"message": "which one?", "extra": 1}))
    decision, _ = await run(case, world, first_turn_only(call))
    assert decision.user_messages == [] and decision.clarified is False


async def test_unknown_tool_uses_production_error_shape():
    world, case = setup()
    decision, captured = await run(case, world, first_turn_only(raw_call("launch_rocket", "{}")))
    tool_message = json.loads(captured[1][-1]["content"])
    assert tool_message == {
        "tool": "launch_rocket", "status": "error", "arguments": {}, "error": {"error": "Unknown tool: launch_rocket"},
    }
    assert decision.tool_calls[0]["error"] == "Unknown tool: launch_rocket"


async def test_second_send_to_a_new_name_reuses_it():
    world, case = setup()
    send = lambda: tool_call("send_message_to_agent", agent_name="Fresh agent", instructions="x")  # noqa: E731
    decision, captured = await run(case, world, first_turn_only(send(), send()))
    stubs = [json.loads(m["content"]) for m in captured[1] if m["role"] == "tool"]
    assert [s["result"]["new_agent_created"] for s in stubs] == [True, False]
    assert decision.created_names == ["Fresh agent"]


async def test_final_assistant_text_counts_as_a_user_message():
    world, case = setup("same_name_people")
    llm = ScriptedLLM(lambda messages, sample: {"content": "  Which Alice do you mean?  "})
    decision = await run_routing_case(case, world, BaselineSystem(), llm, model="m", sample=0)
    assert decision.user_messages == ["Which Alice do you mean?"]
    assert decision.clarified is True


async def test_final_text_is_ignored_when_a_user_message_was_sent():
    world, case = setup("same_name_people")
    call = tool_call("send_message_to_user", message="on it")
    decision, _ = await run(case, world, first_turn_only(call, content="anything else?"))
    assert decision.user_messages == ["on it"]
    assert decision.clarified is False


async def test_clarified_needs_the_last_message_to_ask():
    world, case = setup("same_name_people")
    first = tool_call("send_message_to_user", message="which one?")
    second = tool_call("send_message_to_user", message="never mind, I'll wait.")
    decision, _ = await run(case, world, first_turn_only(first, second))
    assert decision.clarified is False


async def test_iteration_cap_is_flagged():
    world, case = setup()
    llm = ScriptedLLM(lambda messages, sample: {"content": "", "tool_calls": [tool_call("wait", reason="x")]})
    capped = await run_routing_case(case, world, BaselineSystem(), llm, model="m", sample=0)
    assert capped.hit_iteration_cap is True
    done = await run_routing_case(
        case, world, BaselineSystem(), ScriptedLLM(lambda m, s: {"content": ""}), model="m", sample=0
    )
    assert done.hit_iteration_cap is False


class MetaLLM:
    def __init__(self, metas):
        self.metas = list(metas)

    async def complete(self, **kwargs):
        meta = self.metas.pop(0)
        calls = [tool_call("wait", reason="x")] if self.metas else []
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": calls}}], "_meta": meta}

    async def aclose(self):
        return None


async def test_fresh_cost_excludes_cached_calls():
    world, case = setup()
    llm = MetaLLM([
        {"cost": 0.02, "cached": True, "cache_key": "a"},
        {"cost": 0.03, "cached": False, "cache_key": "b"},
    ])
    decision = await run_routing_case(case, world, BaselineSystem(), llm, model="m", sample=0)
    assert decision.cost == pytest.approx(0.05)
    assert decision.fresh_cost == pytest.approx(0.03)
    assert decision.cached_calls == 1


def test_render_context_handles_long_contexts():
    from evals.cases import Case
    from evals.layers.routing import render_context

    many = Case(
        id="x", kind="follow_up", world_seed=1, message="m", gold_targets=["a"],
        context=[("user_message", f"t{i}") for i in range(150)],
    )
    rendered = render_context(many)
    assert 'timestamp="2026-10-24 10:00:00"' in rendered
    assert rendered.count("<user_message ") == 150
    stamps = re.findall(r'timestamp="([^"]+)"', rendered)
    parsed = [datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S") for stamp in stamps]
    assert parsed == sorted(parsed) and len(set(parsed)) == 150
