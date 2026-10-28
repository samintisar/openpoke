# Experiment journal

One entry per experiment, written **before** the next experiment starts. Tune on `dev` only.
The `test` split runs once per finalist (`--split test --final`).

Template:

## <date> · <experiment name>
- **Hypothesis:** what should change, and why (cite the error-analysis category).
- **Change:** config / prompt / code diff, one variable.
- **Run:** result file name.
- **Dev result:** pass@1 / pass^k / duplicate rate per roster size, vs the previous best.
- **Regression suite:** pass / fail.
- **Decision:** keep or revert, and why.

---

## 2026-10-28 · baseline
- **Hypothesis:** with the production roster (bare names, every agent listed), routing quality falls as the roster grows, and duplicates appear.
- **Change:** none: the baseline system mirrors production. Model `anthropic/claude-haiku-5.5` (the spec's Sonnet 4 is no longer served; Sonnet 4.5 was too expensive; free models cap at 20 req/min and 1000 req/day, too slow).
- **Run:** `20261028-074635-baseline-dev.json` ($0.64), regraded as `20261028-075013-baseline-dev.json` ($0, all cached) after error analysis found an eval bug: agent-result cases penalised relaying a question to the user (33 of 49 failures). An earlier Sonnet 4.5 attempt ran out of credits mid-run and is not reported.
- **Dev result (regraded), pass@1 / pass^k / duplicate rate:** n=10 0.89 / 0.86 / 0.00 · n=50 0.95 / 0.95 / 0.00 · n=200 0.95 / 0.95 / 0.00 · n=1000 0.97 / 0.90 / 0.00 · handwritten 0.90 / 0.90 / 0.00. Prompt tokens per call 10.2k → 50.7k from n=10 to n=1000.
- **Regression suite:** seeded with the 26 of 31 cases that passed every sample at every size.
- **Decision:** the hypothesis is mostly rejected for accuracy: overload shows up as 5× cost and lower consistency at n=1000, and as name-only identity errors at every size (13 of 18 failures, see `evals/results/error_analysis.md`). Plan 2 targets: agent cards with entities, top-K retrieval to cut tokens, dedup on entities.

## 2026-10-28 · baseline-lived
- **Hypothesis:** on a roster the IA named itself (inconsistent, muddled names, real history), routing quality falls as the roster grows, which the synthetic world could not show.
- **Change:** new benchmark world, not a system change. `evals/worlds/lived/world-s1.json`: 200 agents grown from 789 life events through real IA turns, plus background errands to 1,000; probes are held-out next events. Generation cost about $4.2 (one regrow after the writer was found leaking its rule notes into ~7 growth messages). From here on this is the only reported benchmark.
- **Run:** `20261028-101527-baseline-lived-dev.json` ($1.00).
- **Dev result, pass@1 / pass^k / duplicate rate:** lived-10 0.87 / 0.86 / 0.00 · lived-50 0.93 / 0.90 / 0.06 · lived-200 0.92 / 0.89 / 0.02 · lived-1000 0.85 / 0.71 / 0.11. Prompt tokens per call 13.8k → 56.0k. The 200 and 1000 sizes share probes.
- **Regression suite:** lived cases that passed every sample at every size were added.
- **Decision:** hypothesis confirmed. Overload shows as near-duplicate agents and lower consistency at 1,000 (see `evals/results/error_analysis_lived.md`). Plan 2 targets: top-K card retrieval, an entity-aware dedup guard, cards with purpose and entities, fuzzy names, ask-on-ambiguity. 7 of 37 failures are label noise and are tracked, not removed.
