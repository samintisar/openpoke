import json

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
