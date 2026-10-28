import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime

import pytest

from evals.grading import grade
from evals.layers.retrieval import run_retrieval_case
from evals.layers.routing import MAX_ITERATIONS, build_messages, run_routing_case
from evals.llm import ScriptedLLM, cache_key, tool_call
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


def test_iteration_cap_matches_production():
    from server.agents.interaction_agent.runtime import InteractionAgentRuntime

    assert MAX_ITERATIONS == InteractionAgentRuntime.MAX_TOOL_ITERATIONS == 8


@pytest.mark.parametrize(
    "name, args",
    [
        ("send_message_to_agent", {"agent_name": "x"}),
        ("send_message_to_agent", {}),
        ("send_message_to_agent", {"agent_name": "x", "instructions": "y", "priority": "high"}),
        ("send_message_to_user", {}),
        ("send_message_to_user", {"message": "m", "extra": 1}),
        ("send_draft", {"to": "a"}),
        ("send_draft", {"to": "a", "subject": "s", "body": "b", "cc": "c"}),
        ("wait", {}),
        ("wait", {"reason": "r", "why": "w"}),
    ],
)
async def test_argument_errors_are_byte_identical_to_production(name, args):
    from server.agents.interaction_agent.tools import handle_tool_call

    expected = handle_tool_call(name, json.dumps(args)).payload["error"]
    world, case = setup()
    decision, captured = await run(case, world, first_turn_only(raw_call(name, json.dumps(args))))
    assert decision.tool_calls[0]["error"] == expected
    assert json.loads(captured[1][-1]["content"])["error"] == {"error": expected}


class FailingLLM:
    """One successful paid call with a tool call, then the given exception."""

    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        if self.calls > 1:
            raise self.exc
        meta = {"cost": 0.04, "cached": False, "cache_key": "k1", "prompt_tokens": 10}
        message = {"role": "assistant", "content": "", "tool_calls": [tool_call("wait", reason="x")]}
        return {"choices": [{"message": message}], "_meta": meta}

    async def aclose(self):
        return None


async def test_mid_case_error_carries_partial_cost():
    from evals.layers.routing import RoutingCaseError

    world, case = setup()
    original = RuntimeError("upstream broke")
    with pytest.raises(RoutingCaseError) as info:
        await run_routing_case(case, world, BaselineSystem(), FailingLLM(original), model="m", sample=0)
    assert info.value.__cause__ is original
    assert "upstream broke" in str(info.value)
    partial = info.value.decision
    assert partial.cost == pytest.approx(0.04) and partial.fresh_cost == pytest.approx(0.04)
    assert partial.iterations == 1 and partial.cache_keys == ["k1"]


async def test_budget_exceeded_propagates_with_partial_decision():
    from evals.llm import BudgetExceeded

    world, case = setup()
    with pytest.raises(BudgetExceeded) as info:
        await run_routing_case(case, world, BaselineSystem(), FailingLLM(BudgetExceeded("cap")), model="m", sample=0)
    assert info.value.decision.fresh_cost == pytest.approx(0.04)


class KeyedLLM:
    """Records the cache key of every payload a ScriptedLLM is sent."""

    def __init__(self, script):
        self.inner = ScriptedLLM(script)
        self.keys = []

    async def complete(self, **kwargs):
        self.keys.append(cache_key(**kwargs))
        return await self.inner.complete(**kwargs)

    async def aclose(self):
        return None


def mixed_script(world, case):
    """Acknowledge plus a bad call, then reuse, create, draft and wait, then finish with text."""
    by_id = world.by_id()
    target = by_id[case.gold_targets[0]].name if case.gold_targets and case.gold_targets[0] in by_id else "Brand new"

    def script(messages, sample):
        seen = tool_turns(messages)
        if seen == 0:
            return {"content": "", "tool_calls": [
                tool_call("send_message_to_user", message="on it?"),
                {"id": "raw", "type": "function", "function": {"name": "send_message_to_agent", "arguments": '{"agent_name": "x"}'}},
            ]}
        if seen == 2:
            return {"content": "", "tool_calls": [
                tool_call("send_message_to_agent", agent_name=target, instructions="do"),
                tool_call("send_message_to_agent", agent_name="Fresh one", instructions="x"),
                tool_call("send_draft", to="a", subject="s", body="b"),
                tool_call("wait", reason="r"),
            ]}
        return {"content": "done"}

    return script


# Decisions and payload cache keys for these runs, as produced by the routing layer at fa4bf6f, before
# systems supplied the prompt, tools and handler. Equal hashes mean the baseline's payloads are unchanged.
BASELINE_GOLDEN_SHA256 = "b8815b161431bbdcc707c84579a137d87cc96505a79fa35506b14560023b44eb"


async def test_baseline_decisions_and_payloads_match_pre_refactor():
    full = build_world(1)
    world = full.truncated(20)
    out = []
    for case in generate_cases(full, per_kind=1)[:8]:
        llm = KeyedLLM(mixed_script(world, case))
        decision = await run_routing_case(case, world, BaselineSystem(), llm, model="m", sample=0)
        out.append({"case": case.id, "decision": asdict(decision), "keys": llm.keys})
    blob = json.dumps(out, indent=1, sort_keys=True).encode("utf-8")
    assert hashlib.sha256(blob).hexdigest() == BASELINE_GOLDEN_SHA256


class SpySystem(BaselineSystem):
    """A baseline with its own prompt, tools, shims and handler, to show the layer uses them."""

    def system_prompt(self):
        return "spy prompt"

    def tool_schemas(self):
        return [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}]

    def tool_signatures(self):
        def ping(target: str) -> None:
            return None

        return {"ping": ping}

    def new_turn(self, world, case):
        return self

    def handle(self, name, args, decision):
        decision.dispatched_ids.append(args["target"])
        return {"tool": name, "status": "success", "arguments": args, "result": {"pong": True}}


async def test_layer_uses_the_systems_prompt_tools_shims_and_handler():
    world, case = setup()
    seen = {}

    class Capture:
        async def complete(self, *, system, tools, messages, **kwargs):
            seen.setdefault("system", system)
            seen.setdefault("tools", tools)
            calls = []
            if tool_turns(messages) == 0:
                calls = [tool_call("ping", target="a1"), raw_call("ping", "{}"), raw_call("send_message_to_agent", "{}")]
            else:
                seen["results"] = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": calls}}], "_meta": {}}

        async def aclose(self):
            return None

    decision = await run_routing_case(case, world, SpySystem(), Capture(), model="m", sample=0)
    assert seen["system"] == "spy prompt" and seen["tools"][0]["function"]["name"] == "ping"
    assert decision.dispatched_ids == ["a1"]
    assert seen["results"][0]["result"] == {"pong": True}
    assert seen["results"][1]["error"]["error"].startswith("Missing required arguments: ")
    assert seen["results"][2]["error"]["error"] == "Unknown tool: send_message_to_agent"
