"""Charts and confidence intervals for the write-up, from committed result files. No plotting dependency: writes SVG.

    python -m evals.charts                      # the final test runs named below -> docs/figures/routing-test.svg

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
SIZES = ("lived-50", "lived-200", "lived-1000")
COLORS = {"Original OpenPoke": "#9aa4b1", "Full cards": "#d9a441", "Retrieval (this work)": "#2f6fdf"}


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


def summary(runs: dict[str, str] = FINAL) -> dict[str, dict[str, tuple[float, float, float, int]]]:
    """system -> size -> (pass@1, low, high, cases); sizes a run doesn't cover are left out."""
    out: dict[str, dict[str, tuple[float, float, float, int]]] = {}
    for label, name in runs.items():
        rows = json.loads((RESULTS / name).read_text(encoding="utf-8"))["routing_rows"]
        out[label] = {}
        for size in SIZES:
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


def main() -> None:
    data = summary()
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "routing-test.svg").write_text(routing_svg(data), encoding="utf-8")
    for label, sizes in data.items():
        for size, (mean, low, high, n) in sizes.items():
            print(f"{label:24} {size:11} {mean:.2f} [{low:.2f}, {high:.2f}]  cases={n}")
    for size in SIZES:
        mean, low, high = paired_difference(FINAL["Retrieval (this work)"], FINAL["Original OpenPoke"], size)
        print(f"retrieval - original     {size:11} {mean:+.2f} [{low:+.2f}, {high:+.2f}]")
    print(f"wrote {FIGURES / 'routing-test.svg'}")


if __name__ == "__main__":
    main()
