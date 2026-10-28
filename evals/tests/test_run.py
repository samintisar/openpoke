import json

import pytest

from evals.grading import ERROR
from evals.leaderboard import write_leaderboard
from evals.llm import Budget, BudgetExceeded, fake_llm
from evals.run import Config, load_config, main, run_experiment
from evals.view import render

METRIC_KEYS = {
    "n_agents", "world_seed", "case_id", "kind", "sample", "outcome", "expects_reuse", "duplicate_possible",
    "prompt_tokens", "cost", "fresh_cost", "cached_calls", "iterations", "hit_iteration_cap", "latency_s",
}


def tiny(name="t", **overrides):
    base = dict(name=name, layers=["routing"], n_agents=[10], k=1, cases_per_kind=1, include_handwritten=False)
    return Config(**{**base, **overrides})


class PaidLLM:
    """fake_llm behaviour, but every call is uncached, costs `cost` and is charged to a Budget."""

    def __init__(self, cap, cost=0.01, fail_on_iteration=None):
        self.inner = fake_llm()
        self.budget = Budget(cap)
        self.cost = cost
        self.fail_on_iteration = fail_on_iteration

    async def complete(self, **kwargs):
        iteration = 1 + sum(1 for m in kwargs["messages"] if m["role"] == "assistant")
        if iteration == self.fail_on_iteration:
            raise RuntimeError("upstream exploded")
        self.budget.check()
        self.budget.add(self.cost)
        data = await self.inner.complete(**kwargs)
        data["_meta"] = {**data["_meta"], "cost": self.cost, "cache_key": f"k{iteration}"}
        return data

    async def aclose(self):
        return None


async def test_smoke_run_writes_results_leaderboard_and_report(tmp_path):
    cfg = Config(name="smoke", n_agents=[10, 20], world_seeds=[1], k=2, cases_per_kind=1, budget_usd=1.0)
    path = await run_experiment(cfg, llm=fake_llm(), out_dir=tmp_path, split="dev", final=False)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert {"10", "20"} <= set(data["summary"]["routing"])
    assert {"10", "20"} <= set(data["summary"]["retrieval"])
    assert data["routing_rows"] and {r["sample"] for r in data["routing_rows"]} == {0, 1}
    assert data["meta"]["checksum"] and data["meta"]["valid"] is True
    assert path.stem in (tmp_path / "leaderboard.md").read_text(encoding="utf-8")

    html = render(data)
    assert "<details>" in html and data["routing_rows"][0]["case_id"] in html


def test_test_split_requires_final(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("name: x\n", encoding="utf-8")
    assert main(["--config", str(config), "--split", "test", "--fake-llm", "--out-dir", str(tmp_path)]) == 2


def test_unknown_config_key_is_rejected(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("name: x\nbogus: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bogus"):
        load_config(config)


def test_cli_fake_run(tmp_path, capsys):
    config = tmp_path / "c.yaml"
    config.write_text("name: cli\nn_agents: [10]\nk: 1\ncases_per_kind: 1\ninclude_handwritten: false\n", encoding="utf-8")
    assert main(["--config", str(config), "--fake-llm", "--out-dir", str(tmp_path)]) == 0
    assert list(tmp_path.glob("*-cli-dev.json"))
    out = capsys.readouterr().out
    assert "errors=0" in out and "hit_cap=0.00" in out and "incomplete=0" in out


async def test_routing_rows_carry_every_metric_and_viewer_key(tmp_path):
    path = await run_experiment(tiny(), llm=fake_llm(), out_dir=tmp_path, split="dev", final=False)
    rows = json.loads(path.read_text(encoding="utf-8"))["routing_rows"]
    assert rows
    for row in rows:
        assert METRIC_KEYS <= set(row)
        assert {"gold", "acceptable", "dispatched_ids", "created_names", "user_messages", "tool_calls",
                "cache_keys", "message", "error"} <= set(row)
        assert row["error"] is None
    assert any(row["duplicate_possible"] for row in rows) and not all(row["duplicate_possible"] for row in rows)


async def test_failed_case_is_an_error_row_with_its_partial_cost(tmp_path):
    llm = PaidLLM(cap=10.0, cost=0.01, fail_on_iteration=2)
    path = await run_experiment(tiny(), llm=llm, out_dir=tmp_path, split="dev", final=False)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["routing_rows"]
    assert rows and all(row["outcome"] == ERROR for row in rows)
    for row in rows:
        assert row["cost"] == pytest.approx(0.01) and row["fresh_cost"] == pytest.approx(0.01)
        assert row["iterations"] == 1 and row["cache_keys"] == ["k1"]
        assert "upstream exploded" in row["error"]
    meta = data["meta"]
    assert meta["fresh_cost_usd"] == pytest.approx(0.01 * len(rows))
    assert meta["unattributed_cost_usd"] == pytest.approx(0.0)
    group = data["summary"]["routing"]["10"]
    assert group["pass_at_1"] is None and group["error_rate"] == 1.0

    html = render(data)
    assert "upstream exploded" in html
    board = (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    assert "– / – / –" in board


async def test_budget_exhaustion_leaves_unrun_cases_out(tmp_path):
    cfg = tiny(concurrency=1)
    # Each case costs two calls; the cap allows the first three calls, so case 1 completes,
    # case 2 dies mid-flight, and the rest never start.
    llm = PaidLLM(cap=0.025, cost=0.01)
    path = await run_experiment(cfg, llm=llm, out_dir=tmp_path, split="dev", final=False)
    data = json.loads(path.read_text(encoding="utf-8"))
    meta, rows = data["meta"], data["routing_rows"]
    assert meta["budget_exhausted"] is True
    assert len(rows) == 1 and rows[0]["outcome"] != ERROR
    assert meta["budget_spent_usd"] == pytest.approx(0.03)
    assert meta["fresh_cost_usd"] == pytest.approx(0.02)
    assert meta["unattributed_cost_usd"] == pytest.approx(0.01)  # the in-flight case's first call
    group = data["summary"]["routing"]["10"]
    assert group["error_count"] == 0
    assert group["planned_cases"] > 1 and group["incomplete_cases"] == group["planned_cases"] - 1


async def test_planned_size_with_no_rows_still_reports(tmp_path, capsys):
    cfg = tiny(n_agents=[10, 20], concurrency=1)
    llm = PaidLLM(cap=0.0)
    path = await run_experiment(cfg, llm=llm, out_dir=tmp_path, split="dev", final=False)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["routing_rows"] == []
    routing = data["summary"]["routing"]
    assert set(routing) == {"10", "20"}
    assert routing["20"]["pass_at_1"] is None and routing["20"]["incomplete_cases"] == routing["20"]["planned_cases"]
    render(data)
    write_leaderboard(tmp_path)


def test_leaderboard_formats_none_rates(tmp_path):
    group = {
        "pass_at_1": None, "pass_hat_k": None, "duplicate_rate": None, "total_cost_usd": 0.0,
        "fresh_cost_usd": 0.0, "error_count": 2, "incomplete_cases": 1,
    }
    data = {"meta": {"system": "baseline", "split": "dev", "valid": True}, "summary": {"routing": {"10": group}}}
    (tmp_path / "r.json").write_text(json.dumps(data), encoding="utf-8")
    board = write_leaderboard(tmp_path).read_text(encoding="utf-8")
    assert "| r | baseline | dev | yes | – / – / – |" in board


async def test_budget_exceeded_on_the_client_is_not_an_error(tmp_path):
    class Broke:
        budget = Budget(0.0)

        async def complete(self, **kwargs):
            raise BudgetExceeded("cap")

        async def aclose(self):
            return None

    path = await run_experiment(tiny(), llm=Broke(), out_dir=tmp_path, split="dev", final=False)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["budget_exhausted"] is True and data["routing_rows"] == []
