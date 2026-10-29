# Plan 6a: Eval Model Switch, Cache Accounting and Cost Guard

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every live eval role from Claude Haiku 5.5 to DeepSeek V4.1 Flash, safely and measurably. This plan:
- records how many prompt tokens were served from the provider's cache;
- refuses accidentally huge prompts;
- passes a sanity gate before any v2 work spends money.

**Architecture:**
- **Cache accounting.** OpenRouter caches DeepSeek prompt prefixes automatically, with no `cache_control` needed. It reports hits in `usage.prompt_tokens_details.cached_tokens`. The eval client (`evals/llm.py`) copies that into each response's `_meta`. Routing decisions sum it, and result rows carry it.
- **Cost guard.** The client estimates prompt size before sending and raises `LongPromptRefused` above 100k tokens unless the run's config sets `allow_long_prompts: true`.
- **Sanity gate.** A small script (`evals/model_gate.py`) reads one routing result and one L3 result and checks three thresholds from spec §5.

**Tech Stack:** Python 3.10+, httpx, pytest (asyncio auto mode), PyYAML, OpenRouter.

**Spec:** `docs/superpowers/specs/2026-10-29-agent-overload-v2-design.md` §5 (Model and cost) and §7 step 1–2.

**Protected files:** `evals/llm.py`, `evals/run.py`, `evals/grading.py`, `evals/layers/routing.py` and `evals/metrics.py` are under the integrity checksum. A live run against an uncommitted change is invalid. Task 5 refreshes the checksum and commits **before** any live run (Task 6).

**Run tests with:** `.venv/Scripts/python -m pytest -q` (656 passing at the start).

---

## File map

| File | Change |
|---|---|
| `evals/llm.py` | `_meta["cached_tokens"]`; DeepSeek fallback price; `LongPromptRefused`; `max_prompt_tokens` on `OpenRouterLLM` |
| `evals/grading.py` | `Decision.cached_prompt_tokens` |
| `evals/layers/routing.py` | sum `cached_tokens` into the decision |
| `evals/run.py` | row field `cached_prompt_tokens`; config `allow_long_prompts`; pass the limit to the client |
| `evals/e2e_run.py`, `evals/memory_run.py` | config `allow_long_prompts`; pass the limit to the client |
| `evals/experiments/*.yaml` (all live configs except judge) | `model: deepseek/deepseek-v4.1-flash` |
| `evals/experiments/model-gate.yaml`, `evals/experiments/model-gate-e2e.yaml` | new: small gate runs |
| `evals/model_gate.py` | new: gate checks over result files |
| `evals/tests/test_llm.py`, `test_layers.py`, `test_run.py`, `test_e2e.py`, `test_memory_eval.py` | tests |
| `evals/tests/test_model_gate.py` | new |
| `evals/experiments/JOURNAL.md` | gate result |

---

### Task 1: Record cached prompt tokens and price DeepSeek in the eval client

**Files:**
- Modify: `evals/llm.py` (`FALLBACK_PRICING`; `OpenRouterLLM.complete`, the `data["_meta"]` block; `_ZERO_META`)
- Test: `evals/tests/test_llm.py`

- [ ] **Step 1: Write the failing tests** (append to `evals/tests/test_llm.py`)

```python
async def test_cached_prompt_tokens_are_recorded(tmp_path):
    body = _ok()
    body["usage"]["prompt_tokens_details"] = {"cached_tokens": 64}
    llm = _llm(tmp_path, lambda request: httpx.Response(200, json=body))
    first = await llm.complete(**KW)
    assert first["_meta"]["cached_tokens"] == 64
    again = await llm.complete(**KW)  # served from our own response cache: the stored meta is kept
    assert again["_meta"]["cached"] is True and again["_meta"]["cached_tokens"] == 64


async def test_cached_tokens_default_to_zero(tmp_path):
    llm = _llm(tmp_path, lambda request: httpx.Response(200, json=_ok()))
    assert (await llm.complete(**KW))["_meta"]["cached_tokens"] == 0


def test_deepseek_fallback_price():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    assert estimate_cost("deepseek/deepseek-v4.1-flash", usage) == pytest.approx(0.045 + 0.50)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest -q evals/tests/test_llm.py -k "cached_prompt or cached_tokens_default or deepseek_fallback"`

Expected: 3 failed (`KeyError: 'cached_tokens'`, and the DeepSeek cost falls back to $3 / $15).

- [ ] **Step 3: Implement**

In `evals/llm.py`, add the DeepSeek price to `FALLBACK_PRICING`:

```python
FALLBACK_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4": (3.0, 15.0),
    "anthropic/claude-sonnet-4.5": (3.0, 15.0),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "deepseek/deepseek-v4.1-flash": (0.045, 0.50),
}
```

Add a helper above `class OpenRouterLLM`:

```python
def _cached_tokens(usage: dict[str, Any]) -> int:
    """Prompt tokens the provider served from its prompt cache (OpenRouter: prompt_tokens_details.cached_tokens)."""
    details = usage.get("prompt_tokens_details") or {}
    try:
        return int(details.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        return 0
```

In `OpenRouterLLM.complete`, add the field to the `_meta` dict:

```python
        data["_meta"] = {
            "cost": cost + wasted,  # includes charged attempts that returned unusable bodies
            "latency_s": round(latency, 3),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "cached_tokens": _cached_tokens(usage),
            "cached": False,
            "cache_key": key,
        }
```

Update `_ZERO_META`:

```python
_ZERO_META = {
    "cost": 0.0, "latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
    "cached": False, "cache_key": "",
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest -q evals/tests/test_llm.py`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add evals/llm.py evals/tests/test_llm.py
git commit -m "Eval client: record provider-cached prompt tokens; DeepSeek V4.1 Flash fallback price"
```

---

### Task 2: Carry cached prompt tokens into routing decisions and result rows

**Files:**
- Modify: `evals/grading.py` (`Decision`)
- Modify: `evals/layers/routing.py` (the per-call accounting block near line 136)
- Modify: `evals/run.py` (`_routing_row`)
- Test: `evals/tests/test_layers.py`, `evals/tests/test_run.py`

- [ ] **Step 1: Write the failing test** (append to `evals/tests/test_layers.py`, next to `test_fresh_cost_excludes_cached_calls`, reusing its `MetaLLM`)

```python
async def test_cached_prompt_tokens_are_summed():
    world, case = setup()
    llm = MetaLLM([
        {"cost": 0.01, "cached": False, "cache_key": "a", "prompt_tokens": 1000, "cached_tokens": 800},
        {"cost": 0.01, "cached": False, "cache_key": "b", "prompt_tokens": 1200, "cached_tokens": 1000},
    ])
    decision = await run_routing_case(case, world, BaselineSystem(), llm, model="m", sample=0)
    assert decision.prompt_tokens == 2200
    assert decision.cached_prompt_tokens == 1800
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest -q evals/tests/test_layers.py -k cached_prompt_tokens_are_summed`

Expected: FAIL with `AttributeError: 'Decision' object has no attribute 'cached_prompt_tokens'`.

- [ ] **Step 3: Implement**

In `evals/grading.py`, in `class Decision`, after `completion_tokens`:

```python
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0  # prompt tokens the provider served from its prompt cache
```

In `evals/layers/routing.py`, after `decision.prompt_tokens += int(meta.get("prompt_tokens", 0))`:

```python
        decision.cached_prompt_tokens += int(meta.get("cached_tokens", 0))
```

In `evals/run.py` `_routing_row`, extend the token fields:

```python
        "cached_calls": d.cached_calls, "latency_s": d.latency_s, "prompt_tokens": d.prompt_tokens,
        "completion_tokens": d.completion_tokens, "cached_prompt_tokens": d.cached_prompt_tokens,
        "cache_keys": d.cache_keys,
```

- [ ] **Step 4: Add a row test** (append to `evals/tests/test_run.py`)

```python
def test_routing_rows_carry_cached_prompt_tokens(tmp_path):
    assert main(["--config", "evals/experiments/smoke.yaml", "--fake-llm", "--out-dir", str(tmp_path)]) == 0
    data = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert all(row["cached_prompt_tokens"] == 0 for row in data["routing_rows"])
```

If `test_run.py` does not already import `json` and `main` (`from evals.run import main`), add those imports at the top.

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python -m pytest -q evals/tests/test_layers.py evals/tests/test_run.py`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add evals/grading.py evals/layers/routing.py evals/run.py evals/tests/test_layers.py evals/tests/test_run.py
git commit -m "Routing rows record provider-cached prompt tokens"
```

---

### Task 3: Refuse prompts over 100k tokens unless a config allows them

**Files:**
- Modify: `evals/llm.py` (new `LongPromptRefused`, `estimate_prompt_tokens`; `OpenRouterLLM.__init__` and `complete`)
- Modify: `evals/run.py` (`Config`, the `OpenRouterLLM(...)` call near line 322)
- Modify: `evals/e2e_run.py` (`E2EConfig`, the `OpenRouterLLM(...)` call near line 195)
- Modify: `evals/memory_run.py` (`MemoryConfig`, the `OpenRouterLLM(...)` call near line 175)
- Test: `evals/tests/test_llm.py`, `evals/tests/test_run.py`

- [ ] **Step 1: Write the failing tests** (append to `evals/tests/test_llm.py`)

```python
from evals.llm import LongPromptRefused, estimate_prompt_tokens


def test_prompt_estimate_counts_system_messages_and_tools():
    payload = build_payload(model="m", system="a" * 400, messages=[{"role": "user", "content": "b" * 400}],
                            tools=[{"type": "function", "function": {"name": "t", "description": "c" * 400}}])
    assert 300 <= estimate_prompt_tokens(payload) <= 400


async def test_long_prompt_is_refused_before_sending(tmp_path):
    sent = []
    llm = OpenRouterLLM(api_key="k", cache_dir=tmp_path, budget=Budget(1.0), max_prompt_tokens=100,
                        transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(200, json=_ok())))
    with pytest.raises(LongPromptRefused, match="allow_long_prompts"):
        await llm.complete(**{**KW, "messages": [{"role": "user", "content": "x" * 2000}]})
    assert sent == []


async def test_no_limit_when_max_prompt_tokens_is_none(tmp_path):
    llm = OpenRouterLLM(api_key="k", cache_dir=tmp_path, budget=Budget(1.0), max_prompt_tokens=None,
                        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_ok())))
    await llm.complete(**{**KW, "messages": [{"role": "user", "content": "x" * 2000}]})


def test_long_prompt_refusal_stops_the_run():
    assert issubclass(LongPromptRefused, RunAborted)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest -q evals/tests/test_llm.py -k "prompt_estimate or long_prompt or no_limit"`

Expected: ImportError (`LongPromptRefused` is not defined).

- [ ] **Step 3: Implement in `evals/llm.py`**

After `class DailyQuotaExhausted`:

```python
class LongPromptRefused(RunAborted):
    """A request's estimated prompt is over the run's limit. Long prompts are priced higher by several
    providers (Haiku 5x above 100k tokens), so a run that needs them must say so (`allow_long_prompts: true`)."""


DEFAULT_MAX_PROMPT_TOKENS = 100_000


def estimate_prompt_tokens(payload: dict[str, Any]) -> int:
    """A deliberately rough size of the prompt: about 4 characters per token over messages and tools."""
    text = json.dumps(payload.get("messages", []), default=str) + json.dumps(payload.get("tools", []), default=str)
    return len(text) // 4
```

In `OpenRouterLLM.__init__`, add the parameter after `rate_limiter` and store it:

```python
        rate_limiter: RateLimiter | None = None,
        max_prompt_tokens: int | None = DEFAULT_MAX_PROMPT_TOKENS,
    ) -> None:
        ...
        self._rate_limiter = rate_limiter
        self.max_prompt_tokens = max_prompt_tokens
```

In `OpenRouterLLM.complete`, after the cache check returns (only uncached calls are refused), before `async with self._semaphore:`:

```python
        if self.max_prompt_tokens is not None:
            estimate = estimate_prompt_tokens(payload)
            if estimate > self.max_prompt_tokens:
                raise LongPromptRefused(
                    f"prompt of ~{estimate} tokens is over the {self.max_prompt_tokens}-token limit; "
                    "set allow_long_prompts: true in the config to send it"
                )
```

- [ ] **Step 4: Wire the config key into the three runners**

`evals/run.py`: in `Config`, after `lived_path`:

```python
    lived_path: str | None = None
    allow_long_prompts: bool = False  # true: send prompts over ~100k tokens (priced higher by several providers)
```

At the `OpenRouterLLM(` call in `main`, pass:

```python
            max_prompt_tokens=None if cfg.allow_long_prompts else DEFAULT_MAX_PROMPT_TOKENS,
```

and import `DEFAULT_MAX_PROMPT_TOKENS` from `evals.llm` alongside the existing imports.

`evals/e2e_run.py`:
- In `E2EConfig`, after `scenarios`, add `allow_long_prompts: bool = False`.
- At the `OpenRouterLLM(` call, pass the same `max_prompt_tokens=...` expression and add the same import.

`evals/memory_run.py`:
- In `MemoryConfig`, after `max_retries`, add `allow_long_prompts: bool = False`.
- At the `OpenRouterLLM(` call, pass the same `max_prompt_tokens=...` expression and add the same import.

- [ ] **Step 5: Add a config test** (append to `evals/tests/test_run.py`)

```python
def test_allow_long_prompts_is_a_known_config_key(tmp_path):
    from evals.run import load_config
    path = tmp_path / "c.yaml"
    path.write_text("name: x\nallow_long_prompts: true\n", encoding="utf-8")
    assert load_config(path).allow_long_prompts is True
```

- [ ] **Step 6: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add evals/llm.py evals/run.py evals/e2e_run.py evals/memory_run.py evals/tests/test_llm.py evals/tests/test_run.py
git commit -m "Eval client refuses prompts over ~100k tokens unless the config sets allow_long_prompts"
```

---

### Task 4: The model gate checks

**Files:**
- Create: `evals/model_gate.py`
- Test: `evals/tests/test_model_gate.py`

The gate implements spec §5. It fails when any of these is over its threshold:

| Check | Fails above |
|---|---|
| Malformed tool calls (a routing tool call whose `error` starts with `invalid json`, `decoded arguments`, `unsupported argument type`, `Missing required arguments` or `Unknown tool`), as a share of routing tool calls | 2% |
| Turns that loop on `wait` (a routing row with 3 or more `wait` calls, or that hit the iteration cap), as a share of routing rows | 10% |
| Error rows (routing outcome `error`; L3 `status == "error"`), as a share of all rows | 2% |

It also prints:
- routing fresh cost per decision;
- the share of prompt tokens served from the provider cache.

- [ ] **Step 1: Write the failing tests** (`evals/tests/test_model_gate.py`)

```python
import json

from evals.model_gate import check, main


def _routing(rows):
    return {"routing_rows": rows}


def _row(outcome="correct", calls=(), cap=False, fresh=0.001, prompt=1000, cached=500):
    return {"outcome": outcome, "tool_calls": list(calls), "hit_iteration_cap": cap,
            "fresh_cost": fresh, "prompt_tokens": prompt, "cached_prompt_tokens": cached}


def _e2e(statuses):
    return {"scenario_rows": [{"status": s} for s in statuses]}


def test_clean_runs_pass():
    report = check(_routing([_row(calls=[{"name": "send_message_to_agent", "arguments": {}}])] * 50), _e2e(["ok"] * 9))
    assert report["passed"] is True
    assert report["cached_share"] == 0.5 and report["fresh_cost_per_decision"] == 0.001


def test_malformed_tool_calls_fail_the_gate():
    bad = {"name": "send_message_to_agent", "arguments": {}, "error": "invalid json: Expecting value"}
    good = {"name": "send_message_to_agent", "arguments": {}}
    rows = [_row(calls=[bad])] * 3 + [_row(calls=[good])] * 97
    report = check(_routing(rows), _e2e(["ok"]))
    assert report["malformed_rate"] == 0.03 and report["passed"] is False


def test_wait_loops_and_cap_hits_fail_the_gate():
    wait = {"name": "wait", "arguments": {"reason": "x"}}
    rows = [_row(calls=[wait] * 3)] * 6 + [_row(cap=True)] * 5 + [_row()] * 89
    report = check(_routing(rows), _e2e(["ok"]))
    assert report["wait_loop_rate"] == 0.11 and report["passed"] is False


def test_errors_count_across_both_runs():
    report = check(_routing([_row(outcome="error")] + [_row()] * 49), _e2e(["error"] + ["ok"] * 49))
    assert report["error_rate"] == 0.02 and report["passed"] is True  # exactly at the threshold passes
    report = check(_routing([_row(outcome="error")] * 2 + [_row()] * 48), _e2e(["error"] + ["ok"] * 49))
    assert report["passed"] is False


def test_cli_exit_code(tmp_path):
    routing = tmp_path / "r.json"
    e2e = tmp_path / "e.json"
    routing.write_text(json.dumps(_routing([_row()] * 10)), encoding="utf-8")
    e2e.write_text(json.dumps(_e2e(["ok"] * 3)), encoding="utf-8")
    assert main(["--routing", str(routing), "--e2e", str(e2e)]) == 0
    e2e.write_text(json.dumps(_e2e(["error"] * 3)), encoding="utf-8")
    assert main(["--routing", str(routing), "--e2e", str(e2e)]) == 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest -q evals/tests/test_model_gate.py`

Expected: `ModuleNotFoundError: No module named 'evals.model_gate'`.

- [ ] **Step 3: Implement `evals/model_gate.py`**

```python
"""Sanity gate for a new eval model (spec 2026-10-29 v2, §5): does it use our tools reliably?

    python -m evals.model_gate --routing evals/results/<ts>-model-gate-dev.json --e2e evals/results/<ts>-model-gate-e2e-dev.json

Fails (exit 1) when malformed tool calls exceed 2% of routing tool calls, wait loops (3+ waits or an
iteration-cap hit) exceed 10% of routing rows, or error rows exceed 2% of all rows. Also prints fresh
cost per routing decision and the share of prompt tokens served from the provider's cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MAX_MALFORMED = 0.02
MAX_WAIT_LOOPS = 0.10
MAX_ERRORS = 0.02
MALFORMED_PREFIXES = (
    "invalid json", "decoded arguments", "unsupported argument type", "Missing required arguments", "Unknown tool",
)


def _share(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def check(routing: dict[str, Any], e2e: dict[str, Any]) -> dict[str, Any]:
    rows = routing["routing_rows"]
    scenarios = e2e["scenario_rows"]
    calls = [call for row in rows for call in row.get("tool_calls", [])]
    malformed = sum(1 for call in calls if str(call.get("error") or "").startswith(MALFORMED_PREFIXES))
    loops = sum(
        1 for row in rows
        if row.get("hit_iteration_cap") or sum(1 for call in row.get("tool_calls", []) if call.get("name") == "wait") >= 3
    )
    errors = sum(1 for row in rows if row.get("outcome") == "error") + sum(1 for s in scenarios if s.get("status") == "error")
    prompt = sum(int(row.get("prompt_tokens", 0)) for row in rows)
    report = {
        "routing_rows": len(rows),
        "e2e_rows": len(scenarios),
        "malformed_rate": _share(malformed, len(calls)),
        "wait_loop_rate": _share(loops, len(rows)),
        "error_rate": _share(errors, len(rows) + len(scenarios)),
        "fresh_cost_per_decision": round(sum(float(row.get("fresh_cost", 0.0)) for row in rows) / len(rows), 6) if rows else 0.0,
        "cached_share": _share(sum(int(row.get("cached_prompt_tokens", 0)) for row in rows), prompt),
    }
    report["passed"] = (
        report["malformed_rate"] <= MAX_MALFORMED
        and report["wait_loop_rate"] <= MAX_WAIT_LOOPS
        and report["error_rate"] <= MAX_ERRORS
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--routing", type=Path, required=True, help="a routing result file (evals.run)")
    parser.add_argument("--e2e", type=Path, required=True, help="an L3 result file (evals.e2e_run)")
    args = parser.parse_args(argv)
    report = check(
        json.loads(args.routing.read_text(encoding="utf-8")), json.loads(args.e2e.read_text(encoding="utf-8"))
    )
    for key, value in report.items():
        print(f"{key:24} {value}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest -q evals/tests/test_model_gate.py`

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add evals/model_gate.py evals/tests/test_model_gate.py
git commit -m "Model gate: malformed tool calls, wait loops and errors over a routing and an L3 run"
```

---

### Task 5: Switch every live eval config to DeepSeek V4.1 Flash, add the gate configs, refresh the checksum

**Files:**
- Modify: the `model:` line in every live config:
  - `evals/experiments/baseline.yaml`
  - `baseline-lived.yaml`
  - `retrieved-lived.yaml`
  - `retrieved-nofuzzy-lived.yaml`
  - `retrieved-noguard-lived.yaml`
  - `retrieved-nolastrouted-lived.yaml`
  - `retrieved-nolife-lived.yaml`
  - `retrieved-nosearch-lived.yaml`
  - `full-cards-lived.yaml`
  - `e2e.yaml`
  - `e2e-gate-off.yaml`
  - `memory.yaml`
  - `memory-heldout.yaml`
  - `memory-heldout-check.yaml`
- Modify: the default `model` in `evals/run.py` `Config`, `evals/e2e_run.py` `E2EConfig` and `evals/memory_run.py` `MemoryConfig`.
- Leave alone: `smoke.yaml` (offline) and `sweep.yaml` (no model calls), if they have no `model:` key.
- Leave alone: the judge's model in `evals/judge.py`.
- Create: `evals/experiments/model-gate.yaml`, `evals/experiments/model-gate-e2e.yaml`.
- Modify tests:
  - `evals/tests/test_e2e.py` (`test_shipped_configs_load`)
  - `evals/tests/test_memory_eval.py` (`test_config_loads_and_rejects_unknown_keys`)

- [ ] **Step 1: Update the two tests that pin the model**

`evals/tests/test_e2e.py`:

```python
    assert on.model == off.model == "deepseek/deepseek-v4.1-flash"
```

`evals/tests/test_memory_eval.py`:

```python
    assert (cfg.model, cfg.seed, cfg.n_agents, cfg.runs, cfg.budget_tokens) == (
        "deepseek/deepseek-v4.1-flash", 1, 12, 40, 4000
    )
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest -q evals/tests/test_e2e.py::test_shipped_configs_load evals/tests/test_memory_eval.py::test_config_loads_and_rejects_unknown_keys`

Expected: 2 failed (the configs still say Haiku).

- [ ] **Step 3: Switch the configs and defaults**

```bash
for f in evals/experiments/*.yaml; do sed -i 's#^model: anthropic/claude-haiku-5.5#model: deepseek/deepseek-v4.1-flash#' "$f"; done
sed -i 's#model: str = "anthropic/claude-haiku-5.5"#model: str = "deepseek/deepseek-v4.1-flash"#' evals/run.py evals/e2e_run.py evals/memory_run.py
grep -rn "haiku" evals --include=*.py --include=*.yaml | grep -v "/results/\|tests/\|FALLBACK_PRICING\|claude-haiku-5.5\": (0.10"
```

Expected: the final grep prints only lines in `evals/worlds/lived/generate.py` and `evals/worlds/lived/round2.py` (world-growth defaults). Change those two defaults the same way, since no Haiku is used anywhere in v2:

```bash
sed -i 's#anthropic/claude-haiku-5.5#deepseek/deepseek-v4.1-flash#' evals/worlds/lived/generate.py evals/worlds/lived/round2.py
```

- [ ] **Step 4: Create the gate configs**

`evals/experiments/model-gate.yaml`:

```yaml
# DeepSeek V4.1 Flash sanity gate (spec v2 §5): a small lived dev slice, one sample per case.
name: model-gate
system: retrieved
model: deepseek/deepseek-v4.1-flash
world: lived
n_agents: [10, 50, 200]
k: 1
include_handwritten: false
budget_usd: 1.0
concurrency: 4
```

`evals/experiments/model-gate-e2e.yaml`:

```yaml
# DeepSeek V4.1 Flash sanity gate, L3 part: three dev scenarios, one sample each.
name: model-gate-e2e
model: deepseek/deepseek-v4.1-flash
k: 1
safety_gate: true
budget_usd: 0.5
scenarios: [approve-draft-happy, inject-cc-attacker-on-reply, concurrent-results-coalesce]
```

- [ ] **Step 5: Check the gate configs load offline**

```bash
.venv/Scripts/python -m evals.run --config evals/experiments/model-gate.yaml --fake-llm --out-dir evals/.cache/gate-check
.venv/Scripts/python -m evals.e2e_run --config evals/experiments/model-gate-e2e.yaml --fake-llm --out-dir evals/.cache/gate-check
```

Expected: both finish with exit code 0 and write a result file (fake model, no spend).

- [ ] **Step 6: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`

Expected: all pass.

- [ ] **Step 7: Refresh the integrity checksum and commit** (required before any live run)

```bash
.venv/Scripts/python -m evals.integrity --update
git add evals/ && git commit -m "Evals run on DeepSeek V4.1 Flash (no Haiku anywhere); model gate configs; checksum refreshed"
.venv/Scripts/python -m evals.integrity
```

Expected: the last command prints matching `current` and `expected` hashes.

---

### Task 6: Run the gate live and record it

**Files:**
- Modify: `evals/experiments/JOURNAL.md`
- Results: `evals/results/<ts>-model-gate-dev.json`, `evals/results/<ts>-model-gate-e2e-dev.json`

- [ ] **Step 1: Run both gate runs** (live; together under about $0.50)

```bash
.venv/Scripts/python -m evals.run --config evals/experiments/model-gate.yaml
.venv/Scripts/python -m evals.e2e_run --config evals/experiments/model-gate-e2e.yaml
```

Expected: each prints `valid=True` and writes its result file.

- [ ] **Step 2: Run the gate**

```bash
.venv/Scripts/python -m evals.model_gate --routing evals/results/<ts>-model-gate-dev.json --e2e evals/results/<ts>-model-gate-e2e-dev.json
```

Expected: prints the report. Exit 0 means pass, exit 1 means fail.

- [ ] **Step 3: Compare with Haiku on the same slice** (free: the old Haiku responses are cached)

Look up the same case ids in `20261029-022629-retrieved-lived-dev.json` (sample 0 only) and count correct samples per size for both models.

This is a sanity read, not a decision rule. Note it in the journal with the caveat that k=1 is noisy.

- [ ] **Step 4: Journal entry** (append to `evals/experiments/JOURNAL.md`)

```markdown
## 2026-10-29 · DeepSeek V4.1 Flash model gate (Plan 6a)
- **Change:** every live eval role moves from Haiku 5.5 to `deepseek/deepseek-v4.1-flash` ($0.045 / $0.50 per M; cached input $0.006).
- **Runs:** `<routing file>` ($x), `<e2e file>` ($y).
- **Gate:** malformed tool calls <m>% (≤2%), wait loops <w>% (≤10%), errors <e>% (≤2%) → **pass/fail**.
- **Cost:** fresh cost per routing decision $<c> (Haiku on the same slice: $<h>); provider-cached share of prompt tokens <s>%.
- **Sanity read vs Haiku (k=1, same cases):** correct <a>/<n> vs <b>/<n>.
- **Decision:** <proceed to Plan 6b | stop and choose another model with the user>.
```

Fill in every `<...>` from the printed numbers before committing.

- [ ] **Step 5: Commit**

```bash
git add evals/results/ evals/experiments/JOURNAL.md
git commit -m "DeepSeek V4.1 Flash gate: <pass|fail>; cost per decision and cache share recorded"
```

- [ ] **Step 6: Stop and report to the user.** On a gate failure, do **not** start Plan 6b: report the numbers and ask which model to try next. On a pass, report the measured cost per decision and re-estimate the v2 total (spec cost table) before Plan 6b.

---

## Plans that follow (written after this one lands, using what it measures)

- **6b — Benchmark v2:**
  - life 2;
  - matter-based stratified probes;
  - three checkers;
  - the review page;
  - severity scoring;
  - re-baseline on v2 dev.
- **6c — System changes:**
  - living summaries;
  - status policy;
  - second look;
  - conversation-aware retrieval;
  - ablations;
  - removals.
- **6d — Final runs and the updated write-up and prep doc.** Optionally, the IRC check.
