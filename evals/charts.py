"""Charts and confidence intervals for the write-up, from committed result files. No plotting dependency: writes SVG.

    python -m evals.charts                      # the final test runs named below -> docs/figures/routing-test.svg
    python -m evals.charts --runs v2-dev        # benchmark v2, life 1 (dev): intervals only, no figure
    python -m evals.charts --runs v2-steps      # Plan 6c steps, original -> full: rates, tokens, paired differences
    python -m evals.charts --runs v2-test       # Plan 6d sealed life 2: scaling table -> docs/figures/scaling-test.svg

pass@1 per size is the mean over cases of each case's share of correct samples. The 95% interval is a bootstrap over
cases (10,000 resamples, seed 0), since the k samples of one case are not independent. The paired difference
resamples the same case ids for both systems.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"
FIGURES = ROOT / "docs" / "figures"

FINAL = {  # system label -> final test-split run (Plan 5 Task 5)
    "Original OpenPoke": "20261029-213407-baseline-lived-test.json",
    "Full cards": "20261029-213620-full-cards-lived-test.json",
    "Retrieval (this work)": "20261029-212710-retrieved-lived-test.json",
}
V2_DEV = {  # benchmark v2 re-baseline on life 1 (Plan 6b Task 12)
    "Original OpenPoke": "20261030-033628-v2-baseline-dev.json",
    "Full cards": "20261030-040630-v2-full-cards-dev.json",
    "Retrieval (this work)": "20261030-034838-v2-retrieved-dev.json",
}
V2_STEPS = {  # Plan 6c, life 1 dev: each step adds one change to the one before (labels in order)
    "original": "20261030-033628-v2-baseline-dev.json",
    "v1 retrieval": "20261030-034838-v2-retrieved-dev.json",
    "step 1 summaries": "20261030-055120-v2-step1-summaries-dev.json",
    "step 1b + search budget": "20261030-064327-v2-step1b-search-budget-dev.json",
    "step 2b + status rule": "20261030-070834-v2-step2b-status-dispatch-dev.json",
    "step 3 + second look (LLM reviewer)": "20261030-072458-v2-step3-second-look-dev.json",
}
FULL_JEV: str | None = "20261030-082330-v2-full-dev.json"  # the full system with the Jev reviewer (frozen for Plan 6d)
V2_TEST = {  # Plan 6d: the sealed life-2 run, once per system
    "Original OpenPoke": "20261030-092828-v2-test-baseline-test.json",
    "v1 retrieval": "20261030-094241-v2-test-v1-test.json",
    "Final (this work)": "20261030-100407-v2-test-full-test.json",
}
SIZES = ("lived-50", "lived-200", "lived-1000")
V2_SIZES = ("lived-10", *SIZES)
COLORS = {"Original OpenPoke": "#9aa4b1", "Full cards": "#d9a441", "Retrieval (this work)": "#2f6fdf"}
SCALING_COLORS = {"Original OpenPoke": "#9aa4b1", "v1 retrieval": "#d9a441", "Final (this work)": "#2f6fdf"}


def case_scores(rows: list[dict], size: str) -> dict[str, float]:
    by_case: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row["n_agents"] == size:
            by_case[row["case_id"]].append(row["outcome"] == "correct")
    return {case: sum(v) / len(v) for case, v in by_case.items()}


def bootstrap(values: list[float], resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def summary(runs: dict[str, str] = FINAL, sizes: tuple[str, ...] = SIZES) -> dict[str, dict[str, tuple[float, float, float, int]]]:
    """system -> size -> (pass@1, low, high, cases); sizes a run doesn't cover are left out."""
    out: dict[str, dict[str, tuple[float, float, float, int]]] = {}
    for label, name in runs.items():
        rows = json.loads((RESULTS / name).read_text(encoding="utf-8"))["routing_rows"]
        out[label] = {}
        for size in sizes:
            scores = list(case_scores(rows, size).values())
            if scores:
                out[label][size] = (sum(scores) / len(scores), *bootstrap(scores), len(scores))
    return out


def paired_difference(a: str, b: str, size: str) -> tuple[float, float, float]:
    """pass@1 of run `a` minus run `b` on the same cases, with a paired bootstrap interval."""
    sa =case_scores(json.loads((RESULTS / a).read_text(encoding="utf-8"))["routing_rows"], size)
    sb = case_scores(json.loads((RESULTS / b).read_text(encoding="utf-8"))["routing_rows"], size)
    diffs = [sa[c] - sb[c] for c in sorted(sa) if c in sb]
    return (sum(diffs) / len(diffs), *bootstrap(diffs))


def routing_svg(data: dict[str, dict[str, tuple[float, float, float, int]]]) -> str:
    width, height, left, top, bottom = 640, 340, 56, 48, 52
    plot_h = height - top - bottom
    group_w = (width - left - 20) / len(SIZES)
    bar_w = group_w / (len(data) + 1.2)
    y = lambda v: top + plot_h * (1 - (v - 0.5) / 0.5)  # axis from 0.5 to 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="system-ui, sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="22" font-size="14" font-weight="600" fill="#1f2328">Routing pass@1 on the held-out test split (95% CI)</text>',
    ]
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        parts.append(f'<line x1="{left}" x2="{width - 20}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="#e6e8eb"/>')
        parts.append(f'<text x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end" fill="#57606a">{tick:.1f}</text>')
    for g, size in enumerate(SIZES):
        x0 = left + g * group_w + bar_w * 0.6
        for i, (label, sizes) in enumerate(data.items()):
            if size not in sizes:
                continue
            mean, low, high, _ = sizes[size]
            x = x0 + i * bar_w
            parts.append(f'<rect x="{x:.1f}" y="{y(mean):.1f}" width="{bar_w * 0.85:.1f}" height="{y(0.5) - y(mean):.1f}" fill="{COLORS[label]}"/>')
            cx = x + bar_w * 0.425
            parts.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(high):.1f}" y2="{y(max(low, 0.5)):.1f}" stroke="#1f2328" stroke-width="1.2"/>')
            parts.append(f'<text x="{cx:.1f}" y="{y(high) - 5:.1f}" text-anchor="middle" font-size="10" fill="#1f2328">{mean:.2f}</text>')
        parts.append(f'<text x="{left + g * group_w + group_w / 2:.1f}" y="{height - bottom + 18}" text-anchor="middle" fill="#1f2328">{size.split("-")[1]} agents</text>')
    lx = left
    for label, color in COLORS.items():
        parts.append(f'<rect x="{lx}" y="{height - 18}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{lx + 14}" y="{height - 9}" fill="#1f2328">{label}</text>')
        lx += 14 + 7.2 * len(label) + 18
    parts.append("</svg>")
    return "\n".join(parts)


def scaling(runs: dict[str, str] = V2_TEST, sizes: tuple[str, ...] = V2_SIZES) -> dict[str, dict[str, tuple[float, float, float, float, float]]]:
    """system -> size -> (pass@1, low, high, mean prompt tokens, mean latency s)."""
    out: dict[str, dict[str, tuple[float, float, float, float, float]]] = {}
    for label, name in runs.items():
        rows = json.loads((RESULTS / name).read_text(encoding="utf-8"))["routing_rows"]
        out[label] = {}
        for size in sizes:
            sized = [r for r in rows if r["n_agents"] == size]
            if sized:
                scores = list(case_scores(rows, size).values())
                tokens = sum(r["prompt_tokens"] for r in sized) / len(sized)
                latency = sum(r["latency_s"] for r in sized) / len(sized)
                out[label][size] = (sum(scores) / len(scores), *bootstrap(scores), tokens, latency)
    return out


def scaling_svg(data: dict[str, dict[str, tuple[float, float, float, float, float]]]) -> str:
    """Three panels against roster size (log x): pass@1 with 95% CI, prompt tokens, latency."""
    import math

    panel_w, height, top, bottom, gap = 250, 300, 56, 64, 46
    width = 3 * panel_w + 2 * gap + 40
    plot_h = height - top - bottom
    sizes = [s for s in V2_SIZES if any(s in d for d in data.values())]
    counts = [int(s.split("-")[1]) for s in sizes]
    lo, hi = math.log10(counts[0]), math.log10(counts[-1])
    panels = (
        ("pass@1 (95% CI)", 0, (0.6, 1.0), lambda v: f"{v:.1f}", (0.6, 0.7, 0.8, 0.9, 1.0)),
        ("prompt tokens per decision", 3, (0, 40_000), lambda v: f"{v / 1000:.0f}k", (0, 10_000, 20_000, 30_000, 40_000)),
        ("latency per decision (s)", 4, (0, 8), lambda v: f"{v:.0f}", (0, 2, 4, 6, 8)),
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="system-ui, sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="40" y="22" font-size="14" font-weight="600" fill="#1f2328">Scaling on the sealed test life (life 2): original vs v1 vs final</text>',
    ]
    for p, (title, idx, (ymin, ymax), fmt, ticks) in enumerate(panels):
        left = 40 + p * (panel_w + gap)
        x = lambda n: left + panel_w * (math.log10(n) - lo) / (hi - lo)
        y = lambda v: top + plot_h * (1 - (v - ymin) / (ymax - ymin))
        parts.append(f'<text x="{left}" y="{top - 14}" font-weight="600" fill="#1f2328">{title}</text>')
        for tick in ticks:
            parts.append(f'<line x1="{left}" x2="{left + panel_w}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="#e6e8eb"/>')
            parts.append(f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" text-anchor="end" fill="#57606a">{fmt(tick)}</text>')
        for n in counts:
            parts.append(f'<text x="{x(n):.1f}" y="{top + plot_h + 18}" text-anchor="middle" fill="#57606a">{n}</text>')
        parts.append(f'<text x="{left + panel_w / 2}" y="{top + plot_h + 36}" text-anchor="middle" fill="#57606a">agents</text>')
        for label, by_size in data.items():
            color = SCALING_COLORS[label]
            pts = [(x(int(s.split("-")[1])), y(by_size[s][idx]), by_size[s]) for s in sizes if s in by_size]
            parts.append(f'<polyline points="{" ".join(f"{px:.1f},{py:.1f}" for px, py, _ in pts)}" fill="none" stroke="{color}" stroke-width="2"/>')
            for px, py, values in pts:
                if idx == 0:
                    parts.append(f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{y(min(values[2], ymax)):.1f}" y2="{y(max(values[1], ymin)):.1f}" stroke="{color}" stroke-width="1.2"/>')
                parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="{color}"/>')
    lx = 40
    for label, color in SCALING_COLORS.items():
        parts.append(f'<rect x="{lx}" y="{height - 18}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{lx + 14}" y="{height - 9}" fill="#1f2328">{label}</text>')
        lx += 14 + 7.2 * len(label) + 18
    parts.append("</svg>")
    return "\n".join(parts)


def v2_steps_report(steps: dict[str, str] | None = None, sizes: tuple[str, ...] = V2_SIZES) -> str:
    """Per-size pass@1 / harmful / duplicate / tokens with intervals for each step, then paired differences vs the
    previous step and vs the first. Metric logic is evals.compare's (imported here to avoid a cycle)."""
    from .compare import METRICS, _mean, _per_case, paired

    if steps is None:
        steps = dict(V2_STEPS)
        if FULL_JEV:
            steps["full (Jev reviewer)"] = FULL_JEV
    rows = {label: json.loads((RESULTS / name).read_text(encoding="utf-8"))["routing_rows"] for label, name in steps.items()}
    labels = list(steps)
    width = max(len(label) for label in labels)
    lines: list[str] = []
    for size in sizes:
        if not any(r["n_agents"] == size for r in rows[labels[-1]]):
            continue
        lines.append(f"\n{size}")
        lines.append(f"  {'':{width}}  {'pass@1':20} {'harmful':20} {'duplicate':20} tokens")
        for label in labels:
            cells = []
            for metric in METRICS:
                values = list(_per_case(rows[label], size, metric).values())
                cells.append(f"{sum(values) / len(values):.3f} [{bootstrap(values)[0]:.3f},{bootstrap(values)[1]:.3f}]" if values else "n/a")
            tokens = _mean(rows[label], size, lambda r: r["prompt_tokens"])
            lines.append(f"  {label:{width}}  {cells[0]:20} {cells[1]:20} {cells[2]:20} {tokens or 0:,.0f}")
        for title, pairs in (("vs previous step", zip(labels, labels[1:])), ("vs original", ((labels[0], l) for l in labels[1:]))):
            lines.append(f"  paired difference, {title} (new - base):")
            for base, new in pairs:
                cells = []
                for metric in METRICS:
                    mean, low, high, _ = paired(rows[base], rows[new], size, metric)
                    cells.append(f"{mean:+.3f} [{low:+.3f},{high:+.3f}]")
                lines.append(f"  {new:{width}}  {cells[0]:20} {cells[1]:20} {cells[2]:20}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", choices=["final", "v2-dev", "v2-steps", "v2-test"], default="final")
    args = parser.parse_args(argv)
    if args.runs == "v2-steps":
        print(v2_steps_report())
        return
    if args.runs == "v2-test":
        data = scaling()
        for label, by_size in data.items():
            for size, (mean, low, high, tokens, latency) in by_size.items():
                print(f"{label:20} {size:11} pass@1 {mean:.3f} [{low:.3f}, {high:.3f}]  tokens {tokens:>7,.0f}  latency {latency:.2f}s")
        print(v2_steps_report(dict(V2_TEST)))
        FIGURES.mkdir(parents=True, exist_ok=True)
        (FIGURES / "scaling-test.svg").write_text(scaling_svg(data), encoding="utf-8")
        print(f"wrote {FIGURES / 'scaling-test.svg'}")
        return
    runs, sizes = (V2_DEV, V2_SIZES) if args.runs == "v2-dev" else (FINAL, SIZES)
    data = summary(runs, sizes)
    for label, by_size in data.items():
        for size, (mean, low, high, n) in by_size.items():
            print(f"{label:24} {size:11} {mean:.2f} [{low:.2f}, {high:.2f}]  cases={n}")
    for size in sizes:
        mean, low, high = paired_difference(runs["Retrieval (this work)"], runs["Original OpenPoke"], size)
        print(f"retrieval - original     {size:11} {mean:+.2f} [{low:+.2f}, {high:+.2f}]")
    if args.runs == "final":
        FIGURES.mkdir(parents=True, exist_ok=True)
        (FIGURES / "routing-test.svg").write_text(routing_svg(data), encoding="utf-8")
        print(f"wrote {FIGURES / 'routing-test.svg'}")


if __name__ == "__main__":
    main()
