# OpenPoke eval harness

Measures how well the Interaction Agent routes work to execution agents as the roster grows.

## Layers
- **L1 retrieval** (no LLM): is the right agent among what the system shows? recall, MRR, roster tokens.
- **L2 routing** (live): one real Interaction Agent turn with the server's own prompt; tools are
  intercepted and the decision is graded by outcome (reuse / create / clarify).
- **L3 end-to-end** (live): ~25 handwritten multi-step scenarios (`worlds/e2e_scenarios.yaml`) run the real
  Interaction and Execution agents, inbox and runner against an in-memory Gmail (`fakes/gmail.py`). Graded by code
  on the end state: unauthorized sends (judged from approval rows and the trace, not the gate's verdict),
  attempted unauthorized sends (gate blocks), required sends, questions asked, duplicate replies, agent reuse.
  Runs that go off script (a rule declared on the step) are reported apart and left out of pass rates.

## Run
```bash
python -m pytest -q                                                    # unit tests, offline
python -m evals.run --config evals/experiments/smoke.yaml --fake-llm   # offline smoke
python -m evals.run --config evals/experiments/baseline.yaml           # live, needs OPENROUTER_API_KEY in .env
python -m evals.view evals/results/<run>.json                          # HTML report with failing transcripts
python -m evals.e2e_run --config evals/experiments/e2e.yaml             # L3, gate on (e2e-gate-off.yaml: ablation)
python -m evals.e2e_run --config evals/experiments/e2e.yaml --fake-llm  # L3 offline plumbing check
python -m evals.e2e_run --interactive --scenario blank                  # L3 by hand: you type the user's side
```

## Reading results
- Rows whose case failed (API or grading error) are `error` outcomes: they keep the cost they ran
  up and are left out of every pass rate, but counted in `error_count` / `error_rate`.
- If the budget runs out, scheduling stops and `meta.budget_exhausted` is true. Cases that never
  finished have no row and count as `incomplete_cases`. `meta.unattributed_cost_usd` is the spend
  not covered by any row (the case in flight when the cap hit).
- `hit_cap_rate` is the share of non-error samples that hit the 8-iteration cap.

## Rules
- Tune on `dev`. `--split test` requires `--final` and runs once per finalist.
- Protected files (cases, generator, grading, metrics, layers, regression suite) are checksummed;
  a run with a mismatch is marked **invalid**. Update only by deliberate commit:
  `python -m evals.integrity --update`.
- Responses are cached in `evals/.cache/llm/`, keyed by request + sample index. Reruns are free.
- Every experiment gets a `experiments/JOURNAL.md` entry.
