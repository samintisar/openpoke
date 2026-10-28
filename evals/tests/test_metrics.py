import pytest

from evals.metrics import summarize_retrieval, summarize_routing


def row(case_id, sample, outcome, n=10, kind="follow_up", reuse=True, seed=1):
    return {
        "n_agents": n, "world_seed": seed, "case_id": case_id, "kind": kind, "sample": sample,
        "outcome": outcome, "expects_reuse": reuse, "prompt_tokens": 1000, "cost": 0.01, "latency_s": 1.0,
    }


def test_pass_at_1_and_pass_hat_k():
    rows = [
        row("a", 0, "correct"), row("a", 1, "correct"),
        row("b", 0, "correct"), row("b", 1, "duplicate_create"),
    ]
    summary = summarize_routing(rows)["10"]
    assert summary["k"] == 2
    assert summary["pass_at_1"] == pytest.approx(0.75)
    assert summary["pass_hat_k"] == pytest.approx(0.5)
    assert summary["duplicate_rate"] == pytest.approx(0.25)
    assert summary["outcome_rates"] == {"correct": 0.75, "duplicate_create": 0.25}
    assert summary["total_cost_usd"] == pytest.approx(0.04)


def test_groups_by_roster_size_and_kind():
    rows = [row("a", 0, "correct", n=10), row("a", 0, "wrong_reuse", n=1000, kind="typo")]
    summary = summarize_routing(rows)
    assert summary["10"]["pass_at_1"] == 1.0
    assert summary["1000"]["by_kind"] == {"typo": 0.0}


def test_duplicate_rate_is_none_without_reuse_cases():
    assert summarize_routing([row("a", 0, "correct", reuse=False)])["10"]["duplicate_rate"] is None


def test_retrieval_ignores_create_cases_for_recall():
    rows = [
        {"n_agents": 10, "recall": 1.0, "mrr": 0.5, "shown_count": 10, "roster_tokens": 100},
        {"n_agents": 10, "recall": None, "mrr": None, "shown_count": 10, "roster_tokens": 100},
    ]
    summary = summarize_retrieval(rows)["10"]
    assert summary["recall"] == 1.0 and summary["mrr"] == 0.5 and summary["cases"] == 2


def test_macro_pass_at_1_weights_kinds_equally():
    rows = [
        row("a", 0, "correct", kind="follow_up"),
        row("b", 0, "correct", kind="follow_up"),
        row("c", 0, "correct", kind="follow_up"),
        row("d", 0, "wrong_reuse", kind="typo"),
    ]
    summary = summarize_routing(rows)["10"]
    assert summary["pass_at_1"] == pytest.approx(0.75)
    assert summary["macro_pass_at_1"] == pytest.approx(0.5)


def test_macro_pass_at_1_averages_case_level_scores_within_a_kind():
    rows = [
        row("a", 0, "correct", kind="follow_up"), row("a", 1, "correct", kind="follow_up"),
        row("b", 0, "correct", kind="typo"), row("b", 1, "wrong_reuse", kind="typo"),
        row("c", 0, "wrong_reuse", kind="typo"), row("c", 1, "wrong_reuse", kind="typo"),
    ]
    summary = summarize_routing(rows)["10"]
    assert summary["pass_at_1"] == pytest.approx(0.5)
    assert summary["macro_pass_at_1"] == pytest.approx((1.0 + 0.25) / 2)
