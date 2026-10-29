"""Run the L3 end-to-end eval: real interaction and execution agents, fake Gmail, code checks on the end state.

    python -m evals.e2e_run --config evals/experiments/e2e.yaml                 # dev split, live, cached
    python -m evals.e2e_run --config evals/experiments/e2e-gate-off.yaml        # ablation: gate logs and allows
    python -m evals.e2e_run --config evals/experiments/e2e.yaml --fake-llm      # offline plumbing check
    python -m evals.e2e_run --config ... --split test --final                   # held-out, once per finalist
    python -m evals.e2e_run --interactive --scenario approve-draft-happy        # type the user's side yourself
    python -m evals.e2e_run --interactive --scenario blank

Each scenario runs k times (sample 0..k-1), one at a time: a run owns the process-wide stores it patches.
Writes evals/results/<ts>-<name>-<split>.json with the same meta, validity and integrity fields as `evals.run`.
If the budget runs out or OpenRouter refuses the account, scheduling stops: the run in flight and every run not
started are left out of the rows (counted as incomplete), meta.aborted names the reason and the CLI exits 1; a
refused account also makes the run invalid. A run that fails for any other reason is an error row.

Interactive mode is live and uncached (a fresh cache dir per session). Type messages as the user; commands:
`/email <from> <subject> | <body>` delivers an email and raises the watcher alert, `/fire <trigger>` fires a
seeded trigger, `/state` reprints the mailbox, `/quit` ends the session. The transcript is saved to
evals/results/interactive/<ts>-<scenario>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tempfile
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .integrity import compute_checksum, expected_checksum, expected_file_committed
from .layers.e2e import (
    DEFAULT_IDLE_TIMEOUT_S,
    E2ESession,
    Scenario,
    blank_scenario,
    fake_e2e_llm,
    load_scenarios,
    run_scenario,
    summarize_e2e,
)
from .llm import (
    Budget,
    BudgetExceeded,
    DailyQuotaExhausted,
    FatalAPIError,
    LLMClient,
    OpenRouterLLM,
    RateLimiter,
    RunAborted,
)
from .run import CACHE_DIR, DEFAULT_OUT, _api_key, _fmt, _git, _is_int, _usd

INTERACTIVE_OUT = DEFAULT_OUT / "interactive"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "experiments" / "e2e.yaml"


@dataclass
class E2EConfig:
    name: str = "e2e"
    model: str = "anthropic/claude-haiku-5.5"  # interaction and execution agents (and search, compaction)
    k: int = 3
    safety_gate: bool = True  # off: the gate decides and reports but lets every send through (ablation)
    budget_usd: float = 3.0
    debounce_seconds: float = 0.2
    idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_S
    agent_timeout_seconds: float = 90
    concurrency: int = 4  # concurrent LLM requests (agents within one scenario run in parallel)
    requests_per_minute: Optional[int] = None
    max_retries: int = 3
    scenarios: Optional[List[str]] = None  # ids to run; None runs every scenario of the split


def _gate_flag(value: Any) -> bool:
    """YAML reads bare on/off as booleans; accept those and the strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"on", "off"}:
        return value.strip().lower() == "on"
    raise ValueError("config key 'safety_gate': must be on or off")


def load_config(path: Path) -> E2EConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(raw) - {f.name for f in fields(E2EConfig)}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if "safety_gate" in raw:
        raw["safety_gate"] = _gate_flag(raw["safety_gate"])
    cfg = E2EConfig(**raw)
    for key in ("k", "concurrency"):
        value = getattr(cfg, key)
        if not _is_int(value) or value <= 0:
            raise ValueError(f"config key {key!r}: must be a positive int")
    if not _is_int(cfg.max_retries) or cfg.max_retries < 0:
        raise ValueError("config key 'max_retries': must be a non-negative int")
    rpm = cfg.requests_per_minute
    if rpm is not None and (not _is_int(rpm) or rpm <= 0):
        raise ValueError("config key 'requests_per_minute': must be null or a positive int")
    for key in ("budget_usd", "debounce_seconds", "idle_timeout_seconds", "agent_timeout_seconds"):
        value = getattr(cfg, key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"config key {key!r}: must be a positive number")
    if cfg.scenarios is not None and (not isinstance(cfg.scenarios, list) or not all(isinstance(s, str) for s in cfg.scenarios)):
        raise ValueError("config key 'scenarios': must be null or a list of scenario ids")
    return cfg


def select_scenarios(cfg: E2EConfig, split: str, scenarios: Optional[List[Scenario]] = None) -> List[Scenario]:
    every = scenarios if scenarios is not None else load_scenarios()
    if cfg.scenarios is not None:
        known = {scenario.id for scenario in every}
        missing = sorted(set(cfg.scenarios) - known)
        if missing:
            raise ValueError(f"unknown scenario ids in config: {missing}")
    return [s for s in every if s.split == split and (cfg.scenarios is None or s.id in cfg.scenarios)]


async def run_e2e_experiment(
    cfg: E2EConfig,
    *,
    llm: LLMClient,
    out_dir: Path,
    split: str = "dev",
    final: bool = False,
    scenarios: Optional[List[Scenario]] = None,
) -> Path:
    started = datetime.now(timezone.utc)
    checksum, expected = compute_checksum(), expected_checksum()
    chosen = select_scenarios(cfg, split, scenarios)
    rows: List[Dict[str, Any]] = []
    aborted: Optional[RunAborted] = None
    for scenario in chosen:
        for sample in range(cfg.k):
            try:
                rows.append(await run_scenario(
                    scenario, llm, model=cfg.model, sample=sample, gate_on=cfg.safety_gate,
                    debounce_seconds=cfg.debounce_seconds, idle_timeout_seconds=cfg.idle_timeout_seconds,
                    agent_timeout_seconds=cfg.agent_timeout_seconds,
                ))
            except RunAborted as exc:
                aborted = exc  # the run in flight is dropped, not recorded as an error
                break
        if aborted is not None:
            break

    budget = getattr(llm, "budget", None)
    spent = budget.spent_usd if budget is not None else None
    summary = summarize_e2e(rows, planned=[scenario.id for scenario in chosen], k=cfg.k)
    result = {
        "meta": {
            **asdict(cfg),
            "system": "e2e",
            "gate": "on" if cfg.safety_gate else "off",
            "split": split,
            "final": final,
            "scenario_ids": [scenario.id for scenario in chosen],
            "git_sha": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "checksum": checksum,
            "expected_checksum": expected,
            "valid": (
                expected is not None and checksum == expected and expected_file_committed()
                and not isinstance(aborted, FatalAPIError)
            ),
            "budget_spent_usd": None if spent is None else round(spent, 4),
            "fresh_cost_usd": summary["fresh_cost_usd"],
            "unattributed_cost_usd": None if spent is None else round(spent - summary["fresh_cost_usd"], 4),
            "budget_exhausted": isinstance(aborted, BudgetExceeded),
            "aborted": None if aborted is None else f"{type(aborted).__name__}: {aborted}",
        },
        "summary": {"e2e": summary},
        "scenario_rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{started:%Y%m%d-%H%M%S}-{cfg.name}-{split}.json"
    path.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return path


def _live_llm(cfg: E2EConfig, cache_dir: Path) -> Optional[LLMClient]:
    key = _api_key()
    if not key:
        print("OPENROUTER_API_KEY is not set. Add it to .env at the repo root.", file=sys.stderr)
        return None
    limiter = RateLimiter(cfg.requests_per_minute) if cfg.requests_per_minute else None
    return OpenRouterLLM(
        api_key=key, cache_dir=cache_dir, budget=Budget(cfg.budget_usd), concurrency=cfg.concurrency,
        max_retries=cfg.max_retries, rate_limiter=limiter,
    )


def print_summary(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary, meta = data["summary"]["e2e"], data["meta"]
    print(f"gate={meta['gate']}  split={meta['split']}  scenarios={summary['planned_scenarios']}  rows={summary['rows']}  "
          f"graded={summary['graded_rows']}")
    print(f"scenario pass@1={_fmt(summary['pass_at_1'])}  pass^k={_fmt(summary['pass_hat_k'])}  "
          + "  ".join(f"{kind}={_fmt(value)}" for kind, value in summary["by_kind"].items()))
    print("checks: " + "  ".join(f"{name}={_fmt(rate)}" for name, rate in summary["check_pass_rates"].items()))
    print(
        f"unauthorized sends={summary['unauthorized_sends']} of {summary['sends']}  "
        f"rate/run={_fmt(summary['unauthorized_rate_per_run'])}  attempted unauthorized={summary['attempted_unauthorized']} "
        f"(to forbidden recipients {summary['attempted_to_forbidden']})  "
        f"gate disagreements={summary['gate_disagreements']}  ungated sends={summary['ungated_sends']}"
    )
    print(f"off_script={summary['off_script']} {summary['off_script_runs']}  errors={summary['error_count']}  "
          f"incomplete={summary['incomplete_scenarios']}")
    for row in data["scenario_rows"]:
        failed = [name for name, ok in row["checks"].items() if not ok]
        state = "ERROR " + str(row["error"]) if row["status"] == "error" else (
            "OFF-SCRIPT " + str(row["off_script_reason"]) if row["off_script"] else ("pass" if row["success"] else "FAIL " + ",".join(failed))
        )
        print(f"  {row['scenario_id']:<34} #{row['sample']}  {state}  sends={len(row['trace'].get('sends') or [])}  "
              f"unauth={row['unauthorized_sends']}  calls={row['llm_calls']}")
    print(
        f"llm calls={summary['llm_calls']} (cached {summary['cached_calls']}) {summary['calls_by_role']}  "
        f"tokens in={summary['prompt_tokens']} out={summary['completion_tokens']}  cost {_usd(summary['total_cost_usd'])}  "
        f"spent {_usd(meta['budget_spent_usd'])}  budget_exhausted={meta['budget_exhausted']}  valid={meta['valid']}  wrote {path}"
    )
    if meta.get("aborted"):
        print(f"RUN ABORTED: {meta['aborted']}", file=sys.stderr)
        if meta["aborted"].startswith(DailyQuotaExhausted.__name__):
            print("Finished calls are cached: rerun the same command after the quota resets to resume.", file=sys.stderr)
        return 1
    return 0


# -- interactive ------------------------------------------------------------------------------------------------


class _Printer:
    """Prints what changed since the last turn: replies, agent results, approvals, sent mail, gate decisions."""

    def __init__(self, session: E2ESession, color: bool) -> None:
        self.session = session
        self.color = color
        self.sends = self.decisions = 0
        self.printed_messages: Dict[int, int] = {}  # turn index -> messages printed
        self.failed_turns: set[int] = set()
        self.printed_runs: set[int] = set()

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def show(self) -> None:
        recorder, gmail = self.session.recorder, self.session.gmail
        for turn in recorder.turns:
            for message in turn.messages[self.printed_messages.get(turn.index, 0):]:
                head = self._paint(f"[assistant {message['kind']}, turn {turn.index} ({turn.origin})]", "36")
                print(f"{head}\n{message['text']}\n")
            self.printed_messages[turn.index] = len(turn.messages)
            if turn.success is False and turn.index not in self.failed_turns:
                self.failed_turns.add(turn.index)
                print(self._paint(f"[turn {turn.index} failed: {turn.error}]", "31"))
        for run in recorder.runs:
            if run.get("result") and run["index"] not in self.printed_runs:
                self.printed_runs.add(run["index"])
                print(self._paint(f"[agent result] {run['result']}", "33"))
        for decision in gmail.decisions[self.decisions:]:
            line = (f"[gate] {decision['kind']} {decision['gmail_ref']} by {decision['agent_id']} to "
                    f"{', '.join(decision['recipients'])}: {decision['reason']}")
            if not decision["would_allow"]:
                tag = "!! BLOCKED" if decision["enforced"] else "!! WOULD BLOCK (gate off, let through)"
                print(self._paint(f"{tag} {line}", "1;31"))
            else:
                print(self._paint(line, "32"))
        self.decisions = len(gmail.decisions)
        for sent in gmail.sent[self.sends:]:
            ok = sent.audit.get("authorized")
            label = "[sent]" if ok else "!! [sent, UNAUTHORIZED]"
            print(self._paint(f"{label} {sent.kind} to {', '.join(sent.recipients())}: {sent.subject}\n{sent.body}\n",
                              "32" if ok else "1;31"))
        self.sends = len(gmail.sent)
        pending = [a for a in self.session.approvals.with_status(("pending", "approved"))]
        if pending:
            print("[pending approvals] " + "; ".join(
                f"#{a.id} {a.kind} to {', '.join(a.recipients.get('to') or [])} ({a.status}"
                f"{', shown' if a.shown_at else ''})" for a in pending))

    def state(self) -> None:
        gmail = self.session.gmail
        print(f"[drafts] {[(d.id, d.to, d.subject) for d in gmail.drafts.values()]}")
        print(f"[sent] {[(s.index, s.kind, s.recipients(), s.subject, s.audit.get('authorized')) for s in gmail.sent]}")
        print(f"[triggers] {self.session.refs['trigger']}")


def _parse_email_command(text: str) -> Dict[str, Any]:
    """`/email <from> <subject> | <body>`."""
    rest = text[len("/email"):].strip()
    head, _, body = rest.partition("|")
    sender, _, subject = head.strip().partition(" ")
    if not sender or not body.strip():
        raise ValueError("usage: /email <from> <subject> | <body>")
    return {"from": sender, "subject": subject.strip() or "(no subject)", "body": body.strip()}


async def interactive(cfg: E2EConfig, scenario: Scenario, llm: LLMClient, *, out_dir: Path) -> Path:
    started = datetime.now(timezone.utc)
    inputs: List[Dict[str, Any]] = []
    session = E2ESession(scenario, llm, model=cfg.model, gate_on=cfg.safety_gate,
                         debounce_seconds=cfg.debounce_seconds, agent_timeout_seconds=cfg.agent_timeout_seconds)
    trace: Dict[str, Any] = {}
    async with session:
        printer = _Printer(session, color=sys.stdout.isatty())
        print(f"Scenario {scenario.id}: {scenario.description}\nGate {'on' if cfg.safety_gate else 'OFF'}. "
              "Commands: /email <from> <subject> | <body>, /fire <trigger>, /state, /quit")
        if scenario.id != "blank":
            print("Scripted steps (for reference, not run): " + " / ".join(
                str(step.get("user") or next(iter(step))) for step in scenario.steps))
        printer.show()
        step = 0
        try:
            while True:
                try:
                    line = (await asyncio.to_thread(input, "you> ")).strip()
                except EOFError:
                    break
                if not line:
                    continue
                if line == "/quit":
                    break
                if line == "/state":
                    printer.state()
                    continue
                session.recorder.step = step
                try:
                    if line.startswith("/email"):
                        spec = _parse_email_command(line)
                        session.email_arrives(spec)
                        inputs.append({"step": step, "email_arrives": spec})
                    elif line.startswith("/fire"):
                        key = line[len("/fire"):].strip()
                        session.fire_trigger(key)
                        inputs.append({"step": step, "fire_trigger": key})
                    else:
                        session.post_user(line)
                        inputs.append({"step": step, "user": line})
                except (ValueError, KeyError) as exc:
                    print(f"[error] {exc}")
                    continue
                step += 1
                if not await session.wait_idle(cfg.idle_timeout_seconds):
                    print(f"[still busy after {cfg.idle_timeout_seconds:g}s; showing what is done]")
                if session.bridge.aborted is not None:
                    print(f"[aborted: {session.bridge.aborted}]")
                    break
                printer.show()
        except KeyboardInterrupt:
            pass
        finally:
            trace = session.trace()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{started:%Y%m%d-%H%M%S}-{scenario.id}.json"
    path.write_text(json.dumps({
        "meta": {"scenario": scenario.id, "model": cfg.model, "gate": "on" if cfg.safety_gate else "off",
                 "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
                 "budget_spent_usd": getattr(getattr(llm, "budget", None), "spent_usd", None)},
        "inputs": inputs,
        "trace": trace,
    }, indent=1, default=str), encoding="utf-8")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="experiment config (default for --interactive: e2e.yaml)")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--final", action="store_true", help="required to run the held-out test split")
    parser.add_argument("--fake-llm", action="store_true", help="offline scripted model; no API calls")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--interactive", action="store_true", help="type the user's messages yourself (live)")
    parser.add_argument("--scenario", default="blank", help="interactive: scenario id to seed from, or blank")
    args = parser.parse_args(argv)
    logging.getLogger("openpoke.server").setLevel(logging.WARNING)

    if args.interactive:
        cfg = load_config(args.config or DEFAULT_CONFIG)
        if args.scenario == "blank":
            scenario = blank_scenario()
        else:
            found = [s for s in load_scenarios() if s.id == args.scenario]
            if not found:
                print(f"unknown scenario {args.scenario!r}", file=sys.stderr)
                return 2
            scenario = found[0]
        llm: Optional[LLMClient]
        if args.fake_llm:
            llm = fake_e2e_llm()
        else:
            llm = _live_llm(cfg, Path(tempfile.mkdtemp(prefix="e2e-interactive-cache-")))
            if llm is None:
                return 2

        async def go_interactive() -> Path:
            try:
                return await interactive(cfg, scenario, llm, out_dir=args.out_dir or INTERACTIVE_OUT)
            finally:
                await llm.aclose()

        path = asyncio.run(go_interactive())
        print(f"transcript saved to {path}")
        return 0

    if args.config is None:
        parser.error("--config is required (except with --interactive)")
    if args.split == "test" and not args.final:
        print("Refusing to run the test split without --final. Tune on dev; run test once per finalist.", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    if args.fake_llm:
        llm = fake_e2e_llm()
    else:
        llm = _live_llm(cfg, CACHE_DIR)
        if llm is None:
            return 2

    async def go() -> Path:
        try:
            return await run_e2e_experiment(cfg, llm=llm, out_dir=args.out_dir or DEFAULT_OUT, split=args.split,
                                            final=args.final)
        finally:
            await llm.aclose()

    return print_summary(asyncio.run(go()))


if __name__ == "__main__":
    raise SystemExit(main())
