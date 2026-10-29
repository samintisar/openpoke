"""Offline L1 sweep of `RetrievalConfig` over the lived world, dev split only. No LLM, $0.

    python -m evals.sweep --config evals/experiments/sweep.yaml

Every grid point runs the production `select` (lifecycle on, via `RetrievedSystem`) and the L1
layer's `run_retrieval_case` at each roster size, and is compared with the current defaults by the
Plan 5 decision rule: a change must beat the default by more than RECALL_MARGIN recall@shown at
200 or 1,000 agents without adding more than TOKEN_MARGIN prompt tokens there.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from server.services.agents.retrieval import DEFAULT_CONFIG, RetrievalConfig

from .cases import Case, filter_cases
from .layers.retrieval import run_retrieval_case
from .metrics import summarize_retrieval
from .run import DEFAULT_OUT, _git
from .systems.retrieved import RetrievedSystem
from .worlds.lived.store import DEFAULT_PATH as LIVED_PATH
from .worlds.lived.store import load_lived
from .worlds.model import World

WEIGHTS = ("w_kw", "w_ent", "w_rec")
RULE_SIZES = ("lived-200", "lived-1000")
RECALL_MARGIN = 0.01
TOKEN_MARGIN = 0.10


@dataclass
class SweepConfig:
    name: str = "sweep"
    n_agents: list[int] = field(default_factory=lambda: [10, 50, 200, 1000])
    lived_path: str | None = None
    mode: str = "one_at_a_time"  # or cartesian: every combination of grid values and simplex points
    grid: dict[str, list[Any]] = field(default_factory=dict)  # RetrievalConfig field -> values
    weight_step: float | None = 0.1  # simplex grid over (w_kw, w_ent, w_rec); null skips it


def load_config(path: Path) -> SweepConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(raw) - {f.name for f in fields(SweepConfig)}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg = SweepConfig(**raw)
    known = {f.name for f in fields(RetrievalConfig)}
    bad = set(cfg.grid) - known
    if bad:
        raise ValueError(f"grid keys not in RetrievalConfig: {sorted(bad)}")
    if set(cfg.grid) & set(WEIGHTS):
        raise ValueError("sweep the weights with weight_step, not grid")
    if cfg.mode not in ("one_at_a_time", "cartesian"):
        raise ValueError("mode must be one_at_a_time or cartesian")
    return cfg


def normalize_weights(w_kw: float, w_ent: float, w_rec: float) -> dict[str, float]:
    """Scale the three weights to sum to 1 (rounded, so grid points compare and print cleanly)."""
    total = w_kw + w_ent + w_rec
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    kw, ent = round(w_kw / total, 6), round(w_ent / total, 6)
    return {"w_kw": kw, "w_ent": ent, "w_rec": round(1 - kw - ent, 6)}


def simplex(step: float) -> list[dict[str, float]]:
    """Every (w_kw, w_ent, w_rec) on a grid of `step`, each >= 0, summing to 1."""
    n = round(1 / step)
    if n <= 0 or abs(n * step - 1) > 1e-9:
        raise ValueError(f"weight_step {step} must divide 1")
    return [normalize_weights(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)]


def _label(overrides: dict[str, Any]) -> str:
    return ",".join(f"{k}={v}" for k, v in overrides.items()) or "default"


def expand_grid(cfg: SweepConfig, base: RetrievalConfig = DEFAULT_CONFIG) -> list[dict[str, Any]]:
    """Grid points as {"param", "overrides"}; the defaults come first, and no config appears twice."""
    weights = simplex(cfg.weight_step) if cfg.weight_step else []
    if cfg.mode == "cartesian":
        keys = list(cfg.grid)
        combos = [dict(zip(keys, values)) for values in itertools.product(*(cfg.grid[k] for k in keys))]
        candidates = [("all", {**c, **w}) for c in combos for w in (weights or [{}])]
    else:
        candidates = [(key, {key: v}) for key, values in cfg.grid.items() for v in values]
        candidates += [("weights", w) for w in weights]
    points = [{"param": "default", "overrides": {}}]
    seen = {base}
    for param, overrides in candidates:
        rc = replace(base, **overrides)
        if rc in seen:
            continue
        seen.add(rc)
        points.append({"param": param, "overrides": overrides})
    return points


class _SelectOnce(RetrievedSystem):
    """`retrieved` that runs `select` once per (world, case): L1 asks for both the ids and the block."""

    def __init__(self, cfg: RetrievalConfig, cards: dict[int, tuple[World, list]]) -> None:
        super().__init__(cfg)
        self._cards_cache = cards  # the world's cards don't depend on cfg; build them once per sweep
        self._last: tuple[Any, Any, list] | None = None

    def selected(self, world: World, case: Case | None) -> list:
        if self._last is None or self._last[0] is not world or self._last[1] is not case:
            self._last = (world, case, super().selected(world, case))
        return self._last[2]


Unit = tuple[str, World, list[Case]]


def load_units(cfg: SweepConfig) -> list[Unit]:
    """Lived worlds at the configured sizes with their dev cases. The test split is never loaded."""
    wanted = {f"lived-{n}" for n in cfg.n_agents}
    path = Path(cfg.lived_path) if cfg.lived_path else LIVED_PATH
    return [(label, world, filter_cases(cases, split="dev")) for label, world, cases in load_lived(path) if label in wanted]


def evaluate(rc: RetrievalConfig, units: list[Unit], cards: dict[int, tuple[World, list]]) -> dict[str, Any]:
    """L1 summary for one config: per size (summarize_retrieval) plus an overall row."""
    system = _SelectOnce(rc, cards)
    rows = [
        {"n_agents": label, "case_id": case.id, **run_retrieval_case(case, world, system)}
        for label, world, cases in units
        for case in cases
    ]
    recalls = [r["recall"] for r in rows if r["recall"] is not None]
    mrrs = [r["mrr"] for r in rows if r["mrr"] is not None]
    overall = {
        "cases": len(rows),
        "recall": round(mean(recalls), 4) if recalls else None,
        "mrr": round(mean(mrrs), 4) if mrrs else None,
        "mean_shown": round(mean(r["shown_count"] for r in rows), 1),
        "mean_roster_tokens": round(mean(r["roster_tokens"] for r in rows), 1),
    }
    return {"overall": overall, "per_size": summarize_retrieval(rows)}


def passes_rule(point: dict[str, Any], default: dict[str, Any]) -> dict[str, Any]:
    """Decision rule against the defaults: recall gain > RECALL_MARGIN at a rule size, tokens there within TOKEN_MARGIN."""
    reasons = []
    for size in RULE_SIZES:
        got, ref = point["per_size"].get(size), default["per_size"].get(size)
        if not got or not ref or got["recall"] is None or ref["recall"] is None:
            continue
        gain = round(got["recall"] - ref["recall"], 6)  # rounded: 0.92 - 0.90 must not miss by float noise
        tokens = round(got["mean_roster_tokens"] / ref["mean_roster_tokens"] - 1, 6) if ref["mean_roster_tokens"] else 0.0
        if gain > RECALL_MARGIN and tokens <= TOKEN_MARGIN:
            reasons.append({"size": size, "recall_gain": round(gain, 4), "token_change": round(tokens, 4)})
    return {"passes": bool(reasons), "wins": reasons}


def _rule_key(point: dict[str, Any]) -> tuple:
    """Higher is better: recall at the rule sizes, then MRR there, then fewer tokens."""
    sizes = [point["per_size"].get(s) for s in RULE_SIZES]
    sizes = [s for s in sizes if s]
    recall = mean(s["recall"] or 0.0 for s in sizes) if sizes else 0.0
    mrr = mean(s["mrr"] or 0.0 for s in sizes) if sizes else 0.0
    tokens = mean(s["mean_roster_tokens"] for s in sizes) if sizes else 0.0
    return (round(recall, 4), round(mrr, 4), -tokens)


def run_sweep(cfg: SweepConfig, units: list[Unit] | None = None) -> dict[str, Any]:
    """Every grid point's results, the K curve, the best point per parameter and the decision rule."""
    units = units if units is not None else load_units(cfg)
    cards: dict[int, tuple[World, list]] = {}
    points = []
    for point in expand_grid(cfg):
        rc = replace(DEFAULT_CONFIG, **point["overrides"])
        points.append({**point, "label": _label(point["overrides"]), "config": asdict(rc), **evaluate(rc, units, cards)})
    default = points[0]
    for point in points:
        point["rule"] = passes_rule(point, default)

    k_points = sorted(
        (p for p in points if p["param"] in ("default", "k")), key=lambda p: p["config"]["k"]
    )
    k_curve = [
        {
            "k": p["config"]["k"],
            "recall": p["overall"]["recall"],
            "mean_roster_tokens": p["overall"]["mean_roster_tokens"],
            "per_size": {s: {"recall": g["recall"], "mean_roster_tokens": g["mean_roster_tokens"]} for s, g in p["per_size"].items()},
        }
        for p in k_points
    ]
    best = {}
    for param in sorted({p["param"] for p in points} - {"default"}):
        group = [p for p in points if p["param"] == param]
        top = max(group, key=_rule_key)
        best[param] = {"label": top["label"], "beats_default": _rule_key(top) > _rule_key(default), "rule": top["rule"]}
    return {
        "default": default["label"],
        "points": points,
        "k_curve": k_curve,
        "best_per_param": best,
        "passing": [p["label"] for p in points if p["rule"]["passes"]],
    }


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    started = datetime.now(timezone.utc)
    sweep = run_sweep(cfg)
    result = {
        "meta": {
            **asdict(cfg),
            "kind": "sweep",
            "split": "dev",
            "base_config": asdict(DEFAULT_CONFIG),
            "decision_rule": {"sizes": list(RULE_SIZES), "recall_margin": RECALL_MARGIN, "token_margin": TOKEN_MARGIN},
            "git_sha": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "points": len(sweep["points"]),
        },
        **sweep,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{started:%Y%m%d-%H%M%S}-{cfg.name}-dev.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")

    default = sweep["points"][0]
    for size, g in default["per_size"].items():
        print(f"default n={size:>10}  recall={_fmt(g['recall'])}  mrr={_fmt(g['mrr'])}  shown={g['mean_shown']}  tokens={g['mean_roster_tokens']}")
    print("K curve: " + "  ".join(f"k={p['k']}: recall={_fmt(p['recall'])} tokens={p['mean_roster_tokens']}" for p in sweep["k_curve"]))
    for param, b in sweep["best_per_param"].items():
        print(f"best {param}: {b['label']}  beats_default={b['beats_default']}  passes_rule={b['rule']['passes']}")
    print(f"{len(sweep['points'])} points; passing the decision rule: {sweep['passing'] or 'none'}  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
