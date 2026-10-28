import pytest

from evals.metrics import summarize_retrieval, summarize_routing


def row(case_id, sample, outcome, n=10, kind="follow_up", reuse=True, seed=1, dup_possible=None, **extra):
    return {
        "n_agents": n, "world_seed": seed, "case_id": case_id, "kind": kind, "sample": sample,
        "outcome": outcome, "expects_reuse": reuse,
        "duplicate_possible": reuse if dup_possible is None else dup_possible,
        "prompt_tokens": 1000, "cost": 0.01, "fresh_cost": 0.01, "cached_calls": 0, "iterations": 1,
        "latency_s": 1.0, **extra,
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


def test_pass_hat_k_only_counts_complete_cases():
    rows = [
        row("a", 0, "correct"), row("a", 1, "correct"),
        row("b", 0, "wrong_reuse"), row("b", 1, "correct"),
        row("c", 0, "correct"),
    ]
    summary = summarize_routing(rows)["10"]
    assert summary["k"] == 2
    assert summary["incomplete_cases"] == 1
    assert summary["pass_hat_k"] == pytest.approx(0.5)
    assert summary["pass_at_1"] == pytest.approx((1 + 0.5 + 1) / 3, abs=1e-4)


def test_errors_are_excluded_from_pass_rates():
    rows = [
        row("a", 0, "correct"), row("a", 1, "error"),
        row("b", 0, "wrong_reuse"), row("b", 1, "correct"),
        row("c", 0, "error", kind="typo"), row("c", 1, "error", kind="typo"),
    ]
    summary = summarize_routing(rows)["10"]
    assert summary["error_count"] == 3
    assert summary["error_rate"] == pytest.approx(0.5)
    assert summary["cases"] == 2  # case c has only errors and drops out
    assert summary["pass_at_1"] == pytest.approx(0.75)
    assert summary["by_kind"] == {"follow_up": 0.75}
    assert summary["macro_pass_at_1"] == pytest.approx(0.75)
    assert summary["k"] == 2 and summary["incomplete_cases"] == 1
    assert summary["pass_hat_k"] == pytest.approx(0.0)


def test_all_error_group_has_no_pass_rates():
    summary = summarize_routing([row("a", 0, "error")])["10"]
    assert summary["pass_at_1"] is None and summary["pass_hat_k"] is None and summary["error_rate"] == 1.0


def test_by_kind_is_case_level():
    rows = [
        row("a", 0, "correct"), row("a", 1, "correct"), row("a", 2, "correct"),
        row("b", 0, "wrong_reuse"),
    ]
    assert summarize_routing(rows)["10"]["by_kind"] == {"follow_up": 0.5}


def test_duplicate_rate_all_includes_optional_dispatch_cases():
    rows = [
        row("a", 0, "duplicate_create", kind="agent_result", reuse=False, dup_possible=True),
        row("b", 0, "correct", reuse=True),
        row("c", 0, "correct", kind="new_task", reuse=False, dup_possible=False),
    ]
    summary = summarize_routing(rows)["10"]
    assert summary["duplicate_rate"] == pytest.approx(0.0)
    assert summary["duplicate_rate_all"] == pytest.approx(0.5)


def test_fresh_cost_and_cache_hit_rate():
    rows = [
        row("a", 0, "correct", cost=0.04, fresh_cost=0.0, cached_calls=2, iterations=2),
        row("b", 0, "correct", cost=0.06, fresh_cost=0.03, cached_calls=1, iterations=2),
    ]
    summary = summarize_routing(rows)["10"]
    assert summary["total_cost_usd"] == pytest.approx(0.10)
    assert summary["fresh_cost_usd"] == pytest.approx(0.03)
    assert summary["cache_hit_rate"] == pytest.approx(0.75)
