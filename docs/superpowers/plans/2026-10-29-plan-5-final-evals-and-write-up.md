# Plan 5: Ablations, final evals and write-up

**Goal:** turn the dev results into a defensible final answer. That means four things:
- show what each piece of the system contributes;
- calibrate the reply-quality judge against the human labels;
- run the held-out test split once;
- write it up so it can be presented.

**Current state:**
- **Systems:** `baseline` and `retrieved` (with lifecycle) have been measured on lived dev. The `full_cards` dev run was cut off by the out-of-credits error and is invalid.
- **Evals:**
  - L3 dev results exist for gate on and gate off.
  - The memory eval has only its tuning set.
  - 42 L3 transcripts are out for human labelling: `evals/labelling/`, artifact `https://claude.ai/artifact/EbYvw5giYcubVxf7rwKLDV`, collection `labels`.
- **Test splits are untouched:** lived test, the L3 test (5 scenarios) and the handwritten test.

**Rules:**
- Everything up to Task 5 is dev only, one change per experiment, each recorded in `JOURNAL.md` before the next.
- The test split runs once per finalist, in Task 5, with `--final`. Nothing is tuned after it.
- Subagents don't commit or run `evals.integrity --update`.

## Task 1: offline sweeps (L1, $0)

**`evals/sweep.py`:** `python -m evals.sweep --config evals/experiments/sweep.yaml`.
- **What it runs:** L1 retrieval over lived dev at every roster size, for a grid of `RetrievalConfig` values. It uses the production `select` with lifecycle on.
- **Grids:**
  - `k` ∈ {4, 6, 8, 12, 16};
  - `(w_kw, w_ent, w_rec)` over a small simplex grid in steps of 0.05–0.1, summing to 1;
  - `archived_min_score` ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
  - `hot_max` ∈ {3, 5, 8}.
- **Reported per point:**
  - recall@shown, overall and per size;
  - MRR;
  - mean cards shown;
  - estimated prompt tokens, from rendering the cards with the production `render_cards`.
- **Output:** `evals/results/<ts>-sweep-dev.json`, plus a K curve (recall and tokens vs K).
- **Decision rule:** a sweep only changes a default if it beats the current value by more than 0.01 recall@shown at 200 or 1,000 agents without adding more than 10% tokens. Any change is then confirmed with one live dev run in Task 2.
- **Tests:** grid expansion, normalization of the weights, determinism.

## Task 2: live ablations (L2 dev, about $5)

All of these are variants of `retrieved`. Eval-only switches are added to `RetrievedSystem`, with one config per variant.

| Variant | Switched off | Mechanism |
|---|---|---|
| `full_cards` (rerun) | retrieval | the existing system, rerun to completion |
| `retrieved_noguard` | duplicate guard | `dispatch` with the guard bypassed: creates never return `possible_duplicate` |
| `retrieved_nosearch` | `search_agents` | the tool is removed from the schemas and its prompt rule is stripped, using the same prompt-section helper the frozen baseline uses |
| `retrieved_nofuzzy` | typo matching | entity matching is exact only, via a `RetrievalConfig.fuzzy` flag (default on) |
| `retrieved_nolastrouted` | last-routed boost | `last_routed_boost = 0` |

- **What it touches:** each variant changes one thing. The server's defaults are unchanged, and the flags default to production behaviour.
- **Reporting:** per variant and per size: pass@1, pass^k and duplicate rate, plus per-kind deltas against `retrieved`. For example, `noguard` should move duplicates, `nofuzzy` should move `typo`, and `nolastrouted` should move `vague_reference`. Results go in a table in `error_analysis_lived.md`.
- **Removing a component:** it is removed (or listed as unproven) only if its ablation moves nothing beyond seed variance on its own kind and overall.
- **Tests:** each flag switches only its one behaviour. The default configs reproduce `retrieved` exactly, checked by identical cache keys on a scripted run.

## Task 3: judge calibration from the human labels

- **Read the labels:** `evals/labelling/collect.py` reads the `labels` collection (via the controller, using `ArtifactData`, into `evals/labelling/labels.json`, committed).
- **The judge:** `evals/judge.py` asks a different model from the agents' (a non-Anthropic model on OpenRouter, cheap, pinned by exact id) the same six checks the human answered. Each check gets one binary question, with the transcript rendered as the labelling page shows it.
  - **The checks:** draft shown exactly; asked before sending; didn't fall for the email; no internal jargon; one clear reply per turn; user would be satisfied.
  - **Output:** JSON `{check: "pass"|"fail"|"na", reason}`.
- **Calibration:**
  - Agreement per check is computed on the labelled transcripts, with N/A counted as its own class.
  - A check below 85% agreement is rewritten (prompt only), with at most 2 rewrites per check, each in the journal.
  - A check that is still below 85% is reported as human-only and not used on the test split.
  - The code checks get the same treatment where a code check maps to a human one: `no_unauthorized_send` against "asked before sending", and `no_duplicate_replies` against "one clear reply".
- **Reporting:** human pass rates per check, judge–human agreement per check, and code–human agreement.
- **Protection:** add `judge.py` to `PROTECTED` once calibrated, then freeze it.
- **Tests:** offline with `ScriptedLLM`. Cover parsing, agreement maths and N/A handling.

## Task 4: held-out memory set

- **Data:** generate a second memory world (`seed=2`, the same generator) as the held-out set. The 60/60 identifier window was tuned on seed 1.
- **Runs:**
  - Run it once on dev to confirm the setup works: the first 3 agents only, about $0.03.
  - It is then part of the Task 5 test runs.

## Task 5: final test-split runs (once)

The finalists are fixed before this task starts: `retrieved` with any sweep change that survived, the gate on, and the calibrated judge.

**Runs:**
1. **Routing:** `baseline`, `full_cards` and `retrieved` on lived test at 10/50/200/1,000 agents (`--split test --final`), plus the handwritten test cases.
2. **L3 test:** the 5 scenarios, gate on and gate off, k=3. The calibrated judge checks then run on all L3 rows, dev and test.
3. **Memory:** the held-out seed 2.

**Recording and failures:**
- Results are recorded as they come out, with no reruns except for infrastructure failures (402, timeout).
- An infrastructure rerun is logged with the reason.
- **Budget:** about $5. Expect a long wall-clock time for L3, which runs scenario by scenario.

## Task 6: charts and write-up

- **Charts (`evals/charts.py`):** static SVG in `docs/figures/`. Use matplotlib if it's already in the venv; otherwise hand-built SVG, with no new dependency without asking.
  1. pass@1 and duplicate rate vs roster size, for the three systems (test split);
  2. prompt tokens per turn vs roster size;
  3. the ablation deltas;
  4. the K sweep (recall and tokens);
  5. L3, gate on vs off: sends, unauthorized sends and attempts;
  6. memory: compacted vs full recall by fact kind, and history tokens.
- **Write-up (`docs/agent-overload.md`, linked from `README.md`):**
  - the problem;
  - how it was measured (the lived world, L1/L2/L3, memory, the integrity guard, dev/test);
  - the results, using the test-split numbers, with dev in an appendix;
  - what each change does and its evidence, one short section per change, citing the ablation;
  - safety;
  - cost and latency;
  - **known limits:** spec §11 and §12 "Open", restated in plain words;
  - what I'd do next;
  - how to run everything.
- **Small production fix:** check the base64-padding issue the fake Gmail found, in the email cleaner (`server/services/gmail/processing.py`). If unpadded payloads come back empty, pad before decoding, add a test, and note it in the write-up.

## Task 7: interview prep (after the write-up)

A prep document for presenting to Anthony:
- a 10-minute talk track;
- one card per change: what, file, why, evidence, trade-off;
- likely questions with honest answers;
- a demo script: leaderboard, one failure and its fix, interactive L3 trying to trick the assistant into sending.

The format is decided with the user at that point.

**Order:** 1 → 2 → (3 when the labels are in; it doesn't block 2 or 4) → 4 → 5 → 6 → 7.

**Cost:** about $11 in total: Task 2 about $5, Task 3 under $1, Task 5 about $5.

**Cut order if short:** shrink the weight grid in Task 1, drop `nofuzzy` and `nolastrouted` from Task 2, then drop the judge from Task 5 and report human labels only. The test-split routing run and the write-up are non-negotiable.
