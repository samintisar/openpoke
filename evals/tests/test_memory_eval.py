"""Memory eval (L2 memory): generator, grading, the compaction + QA layer and the CLI, all offline."""

import hashlib
import json

import pytest

from evals.layers.memory import (
    QA_PROMPT,
    is_recalled,
    normalize,
    run_memory_agent,
    summarize_memory,
    system_prompt,
)
from evals.leaderboard import write_leaderboard
from evals.llm import BudgetExceeded
from evals.memory_run import MemoryConfig, load_config, main, run_memory_experiment
from evals.view import render
from evals.worlds.memory_gen import FACT_KINDS, MIN_RUNS, build_histories
from server.agents.execution_agent.agent import SYSTEM_PROMPT_TEMPLATE
from server.services.agents.memory import COMPACTION_PROMPT, estimate_tokens, render_entries, split_for_budget

BUDGET = 4000
HISTORIES = build_histories()


def fact(agent, kind):
    return next(f for f in agent.facts if f.kind == kind)


# -- generator ------------------------------------------------------------------------------------------------


def test_generator_is_deterministic_and_seeded():
    assert build_histories() == HISTORIES
    assert build_histories(seed=2) != HISTORIES
    assert len(HISTORIES) == 12 and len({a.agent_id for a in HISTORIES}) == 12
    assert len({a.agent_id for a in build_histories(n_agents=14)}) == 14  # matters past the twelfth are renamed


def test_runs_follow_production_log_order():
    tags = ("agent_request", "agent_action", "tool_response", "agent_response")
    for agent in HISTORIES:
        assert len(agent.entries) == 40 * 4
        assert all(entry.tag == tags[i % 4] for i, entry in enumerate(agent.entries))
        for action, response in zip(agent.entries[1::4], agent.entries[2::4], strict=True):
            tool = action.text.removeprefix("Calling ").split(" with: ")[0]
            assert len(action.text) <= len(f"Calling {tool} with: ") + 200
            assert response.text.startswith(f"{tool}: ") and len(response.text) <= len(f"{tool}: ") + 500


def test_every_agent_has_one_fact_of_each_kind():
    for agent in HISTORIES:
        assert tuple(f.kind for f in agent.facts) == FACT_KINDS
        assert all(f.question and f.gold and f.needle for f in agent.facts)


def test_facts_land_in_the_prefix_and_never_in_the_tail():
    for agent in HISTORIES:
        full = render_entries(agent.entries)
        prefix, tail = split_for_budget(list(agent.entries), BUDGET)
        prefix_text, tail_text = render_entries(prefix), render_entries(tail)
        assert estimate_tokens(full) > 2 * BUDGET  # well over budget
        assert estimate_tokens(prefix_text) > BUDGET  # so production compacts it
        assert len(tail) // 4 >= 10
        for f in agent.facts:
            assert f.run <= 40 // 3
            assert f.needle in prefix_text, (agent.agent_id, f.kind)
            assert f.needle.lower() not in tail_text.lower(), (agent.agent_id, f.kind)
            assert any(normalize(g) in normalize(prefix_text) for g in f.gold)


def test_distractors_appear_but_are_not_the_answer():
    for agent in HISTORIES:
        full = render_entries(agent.entries)
        for kind in ("thread_id", "email", "decision"):
            f = fact(agent, kind)
            assert f.distractors and all(d in full for d in f.distractors)
            assert not any(is_recalled(d, f.gold) for d in f.distractors)


def test_superseded_decision_keeps_the_latest_value():
    for agent in HISTORIES:
        f = fact(agent, "decision")
        (old,) = f.distractors
        prefix, _ = split_for_budget(list(agent.entries), BUDGET)
        text = render_entries(prefix)
        assert text.index(old) < text.index(f.gold[0])  # stated first, then superseded
        assert "Updated:" in text
        assert is_recalled(f"It is now {f.gold[0]}.", f.gold)
        assert not is_recalled(f"We agreed on {old}.", f.gold) and is_recalled(f"We agreed on {old}.", f.distractors)


def test_too_few_runs_are_rejected():
    with pytest.raises(ValueError, match="runs"):
        build_histories(runs=MIN_RUNS - 1)


# Pins the exact histories. Any change to memory_gen.py changes it: update deliberately after checking them.
GOLDEN_SHA256 = "4471901f1441c3351e2b15b56db58a2a50358cdac47a6e369db87b0557163319"


def test_histories_match_the_golden_hash():
    text = "\n".join(
        f"{a.agent_id}|{a.name}|{a.purpose}\n{render_entries(a.entries)}\n{a.facts!r}" for a in HISTORIES
    )
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == GOLDEN_SHA256


# -- grading --------------------------------------------------------------------------------------------------


def test_normalize_strips_case_currency_punctuation_and_space():
    assert normalize("  $4,200.00 / Month ") == "420000month"
    assert normalize("R-8487106398570743728") == "r8487106398570743728"
    assert normalize("D.Okoye20@Whitman-Properties.com") == "dokoye20whitmanpropertiescom"


def test_is_recalled_matches_any_gold_variant():
    assert is_recalled("The agreed rent is $4,200 per month.", ["$4,200"])
    assert is_recalled("4200", ["$4,200"])
    assert not is_recalled("$4,750", ["$4,200"])
    assert is_recalled("By Monday, Sept. 14th.", ["September 14", "Sept 14"])
    assert not is_recalled("September 4", ["September 14"])
    assert is_recalled("Thread 119649BB5F34B724", ["119649bb5f34b724"])
    assert not is_recalled("", ["x"]) and not is_recalled("anything", [""])


# -- layer ----------------------------------------------------------------------------------------------------


class Oracle:
    """Compaction keeps the memory it is given; QA answers a fact's gold iff its needle is in the history."""

    def __init__(self, agent, memory=None, compaction="ok"):
        self.facts = {f.question: f for f in agent.facts}
        self.memory, self.compaction = memory, compaction
        self.calls = []

    async def complete(self, *, model, system, messages, tools, sample):
        self.calls.append({"model": model, "system": system, "messages": messages, "tools": tools, "sample": sample})
        user = messages[-1]["content"]
        if "<new_entries>" in user:
            if self.compaction == "abort":
                raise BudgetExceeded("budget of $0.00 reached")
            content = self.compaction if self.compaction != "ok" else json.dumps(
                {"memory": self.memory, "purpose": "Refreshed purpose", "last_status": "Waiting"}
            )
        else:
            question = user.split(": ", 1)[1]
            f = self.facts[question]
            history = system.split("# Execution History\n\n", 1)[1]
            content = f.gold[0] if f.needle.lower() in history.lower() else "I don't know."
        meta = {"cost": 0.001, "prompt_tokens": len(system) // 4, "completion_tokens": 5, "cached": False,
                "cache_key": str(len(self.calls))}
        return {"choices": [{"message": {"role": "assistant", "content": content}}], "_meta": meta}

    async def aclose(self):
        return None


AGENT = HISTORIES[0]


async def test_layer_compacts_with_production_prompts_and_scores_both_conditions():
    decision, preference = fact(AGENT, "decision"), fact(AGENT, "preference")
    llm = Oracle(AGENT, memory=f"Agreed {decision.gold[0]}. User preference: {preference.needle}.")
    row = await run_memory_agent(AGENT, llm, model="m", budget_tokens=BUDGET)

    compaction_calls = [c for c in llm.calls if "<new_entries>" in c["messages"][-1]["content"]]
    assert len(compaction_calls) == 1
    call = compaction_calls[0]
    assert call["system"] == COMPACTION_PROMPT and call["tools"] == [] and call["sample"] == 0
    assert call["messages"][-1]["content"].startswith(f"Purpose: {AGENT.purpose}")
    qa = [c for c in llm.calls if c not in compaction_calls]
    assert len(qa) == 2 * len(FACT_KINDS)
    assert {c["messages"][0]["content"] for c in qa} == {QA_PROMPT.format(question=f.question) for f in AGENT.facts}
    base = SYSTEM_PROMPT_TEMPLATE.format(agent_name=AGENT.name, agent_purpose=AGENT.purpose)
    assert any(c["system"] == f"{base}\n\n# Execution History\n\n{render_entries(AGENT.entries)}" for c in qa)
    compacted_prompts = [c["system"] for c in qa if "<agent_memory>" in c["system"]]
    assert len(compacted_prompts) == len(FACT_KINDS)
    assert all("Purpose: Refreshed purpose" in p for p in compacted_prompts)  # compaction refreshes the card

    assert row["compaction"]["status"] == "ok" and row["compaction"]["cost"] == pytest.approx(0.001)
    assert row["compacted_history_tokens"] < row["full_history_tokens"]
    full = {a["kind"]: a["correct"] for a in row["conditions"]["full"]["facts"]}
    compacted = {a["kind"]: a["correct"] for a in row["conditions"]["compacted"]["facts"]}
    assert all(full.values())
    # ids and the address survive through <agent_identifiers>; the memory kept the decision and the preference
    assert compacted == {"thread_id": True, "draft_id": True, "email": True, "preference": True, "decision": True,
                         "commitment": False}

    summary = summarize_memory([row], planned=1)
    assert summary["full"]["recall"] == 1.0 and summary["compacted"]["recall"] == pytest.approx(5 / 6, abs=1e-3)
    assert summary["compacted"]["by_kind"]["commitment"] == 0.0 and summary["compacted"]["fallback_agents"] == 0
    assert summary["compaction"] == {**summary["compaction"], "calls": 1, "failures": 0, "skipped": 0}
    assert summary["total_cost_usd"] == pytest.approx(0.013)
    assert summary["full"]["mean_prompt_tokens"] > summary["compacted"]["mean_prompt_tokens"]


async def test_stale_memory_is_flagged():
    decision = fact(AGENT, "decision")

    class StaleOracle(Oracle):
        async def complete(self, **kwargs):
            data = await super().complete(**kwargs)
            if decision.question in kwargs["messages"][-1]["content"] and "<agent_memory>" in kwargs["system"]:
                data["choices"][0]["message"]["content"] = decision.distractors[0]
            return data

    row = await run_memory_agent(AGENT, StaleOracle(AGENT, memory="Something."), model="m", budget_tokens=BUDGET)
    answer = next(a for a in row["conditions"]["compacted"]["facts"] if a["kind"] == "decision")
    assert answer["correct"] is False and answer["stale"] is True


async def test_compaction_failure_falls_back_to_the_full_history():
    llm = Oracle(AGENT, compaction="not json at all")
    row = await run_memory_agent(AGENT, llm, model="m", budget_tokens=BUDGET)
    assert row["compaction"]["status"] == "failed" and "JSON" in row["compaction"]["error"]
    assert row["conditions"]["compacted"]["fallback"] is True
    assert row["conditions"]["compacted"]["facts"] == row["conditions"]["full"]["facts"]
    assert len(llm.calls) == 1 + len(FACT_KINDS)  # no second set of QA calls
    summary = summarize_memory([row], planned=1)
    assert summary["compaction"]["failures"] == 1 and summary["compacted"]["fallback_agents"] == 1
    assert summary["compacted"]["recall"] == summary["full"]["recall"] == 1.0
    assert summary["compacted"]["qa_cost_usd"] == 0.0


async def test_prefix_under_budget_skips_compaction():
    llm = Oracle(AGENT)
    row = await run_memory_agent(AGENT, llm, model="m", budget_tokens=100_000)
    assert row["compaction"]["status"] == "skipped" and row["conditions"]["compacted"]["fallback"] is True
    assert summarize_memory([row], planned=1)["compaction"]["skipped"] == 1


async def test_run_aborted_inside_compaction_is_not_swallowed():
    with pytest.raises(BudgetExceeded):
        await run_memory_agent(AGENT, Oracle(AGENT, compaction="abort"), model="m", budget_tokens=BUDGET)


def test_system_prompt_matches_production_shape():
    prompt = system_prompt("Agent", "Do it", "<x>")
    assert prompt.endswith("\n\n# Execution History\n\n<x>") and "Agent Name: Agent" in prompt


# -- experiment and CLI ---------------------------------------------------------------------------------------


async def test_experiment_aborts_cleanly_on_budget(tmp_path):
    class Broke:
        async def complete(self, **kwargs):
            raise BudgetExceeded("budget of $0.00 reached")

    path = await run_memory_experiment(MemoryConfig(n_agents=3, concurrency=2), llm=Broke(), out_dir=tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["budget_exhausted"] is True and data["meta"]["aborted"].startswith("BudgetExceeded")
    assert data["agent_rows"] == [] and data["summary"]["memory"]["incomplete_agents"] == 3


def test_config_loads_and_rejects_unknown_keys(tmp_path):
    from evals.integrity import REPO_ROOT

    cfg = load_config(REPO_ROOT / "evals" / "experiments" / "memory.yaml")
    assert (cfg.model, cfg.seed, cfg.n_agents, cfg.runs, cfg.budget_tokens) == (
        "anthropic/claude-haiku-5.5", 1, 12, 40, 4000
    )
    bad = tmp_path / "c.yaml"
    bad.write_text("bogus: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bogus"):
        load_config(bad)


def test_cli_fake_run_writes_a_result_the_leaderboard_and_view_tolerate(tmp_path, capsys):
    config = tmp_path / "c.yaml"
    config.write_text("name: memory\nn_agents: 2\n", encoding="utf-8")
    assert main(["--config", str(config), "--fake-llm", "--out-dir", str(tmp_path)]) == 0
    (path,) = tmp_path.glob("*-memory-dev.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    meta, summary = data["meta"], data["summary"]["memory"]
    assert meta["checksum"] and "valid" in meta and meta["budget_tokens"] == 4000 and meta["seed"] == 1
    assert summary["agents"] == 2 and summary["compaction"]["calls"] == 2 and summary["compaction"]["failures"] == 0
    assert all(r["compacted_history_tokens"] < r["full_history_tokens"] for r in data["agent_rows"])
    out = capsys.readouterr().out
    assert "compacted  recall=" in out and "compaction calls=2" in out

    board = write_leaderboard(tmp_path).read_text(encoding="utf-8")
    assert path.stem not in board
    render(data)
