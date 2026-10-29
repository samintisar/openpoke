import itertools
from dataclasses import replace
from pathlib import Path

import pytest

from evals.sweep import (
    SweepConfig,
    expand_grid,
    load_config,
    load_units,
    normalize_weights,
    passes_rule,
    run_sweep,
    simplex,
)
from server.services.agents.retrieval import DEFAULT_CONFIG

REPO = Path(__file__).resolve().parents[2]


def test_normalize_weights_sums_to_one():
    w = normalize_weights(2, 1, 1)
    assert w == {"w_kw": 0.5, "w_ent": 0.25, "w_rec": 0.25}
    for a, b, c in [(0.45, 0.40, 0.15), (1, 1, 1), (0.3, 0.3, 0.3), (7, 0, 0)]:
        assert sum(normalize_weights(a, b, c).values()) == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError):
        normalize_weights(0, 0, 0)


def test_simplex_points_sum_to_one_and_are_unique():
    points = simplex(0.1)
    assert len(points) == 66  # C(12, 2): compositions of 10 into 3 non-negative parts
    assert all(sum(p.values()) == pytest.approx(1.0) and min(p.values()) >= 0 for p in points)
    assert len({tuple(p.values()) for p in points}) == len(points)
    assert len(simplex(0.05)) == 231
    with pytest.raises(ValueError):
        simplex(0.3)


def test_one_at_a_time_grid_varies_one_parameter_and_skips_the_default():
    cfg = SweepConfig(grid={"k": [4, 8, 12], "hot_max": [3, 5]}, weight_step=0.5)
    points = expand_grid(cfg)
    assert points[0] == {"param": "default", "overrides": {}}
    # k=8 and hot_max=5 are the defaults, so they collapse into the first point.
    assert [p["overrides"] for p in points if p["param"] == "k"] == [{"k": 4}, {"k": 12}]
    assert [p["overrides"] for p in points if p["param"] == "hot_max"] == [{"hot_max": 3}]
    assert len([p for p in points if p["param"] == "weights"]) == 6
    assert all(len(p["overrides"]) == 1 for p in points if p["param"] in ("k", "hot_max"))


def test_cartesian_grid_is_the_full_product():
    cfg = SweepConfig(mode="cartesian", grid={"k": [4, 6], "hot_max": [3, 8]}, weight_step=0.5)
    points = expand_grid(cfg)
    assert len(points) == 1 + 2 * 2 * 6
    configs = {replace(DEFAULT_CONFIG, **p["overrides"]) for p in points}
    assert len(configs) == len(points)
    for k, hot in itertools.product([4, 6], [3, 8]):
        assert replace(DEFAULT_CONFIG, k=k, hot_max=hot, w_kw=1.0, w_ent=0.0, w_rec=0.0) in configs


def test_config_rejects_unknown_and_weight_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("grid: {nope: [1]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RetrievalConfig"):
        load_config(bad)
    bad.write_text("grid: {w_kw: [0.5]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="weight_step"):
        load_config(bad)
    cfg = load_config(REPO / "evals" / "experiments" / "sweep.yaml")
    assert cfg.grid["k"] == [4, 6, 8, 12, 16]


def test_units_are_dev_only():
    units = load_units(SweepConfig(n_agents=[10]))
    assert [label for label, _, _ in units] == ["lived-10"]
    assert units[0][2] and all(case.split == "dev" for case in units[0][2])


def test_sweep_is_deterministic():
    cfg = SweepConfig(n_agents=[10, 50], grid={"k": [4, 12]}, weight_step=None)
    units = load_units(cfg)
    first, second = run_sweep(cfg, units), run_sweep(cfg, units)
    assert first == second
    assert [p["label"] for p in first["points"]] == ["default", "k=4", "k=12"]
    assert [p["k"] for p in first["k_curve"]] == [4, 8, 12]


def test_decision_rule_needs_recall_gain_within_token_budget():
    def point(recall, tokens):
        return {"per_size": {"lived-1000": {"recall": recall, "mean_roster_tokens": tokens}}}

    default = point(0.90, 100)
    assert passes_rule(point(0.92, 110), default)["passes"]
    assert not passes_rule(point(0.92, 111), default)["passes"]  # 11% more tokens
    assert not passes_rule(point(0.905, 90), default)["passes"]  # gain not above 0.01
