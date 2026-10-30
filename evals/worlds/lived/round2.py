"""Round-2 probes for the frozen lived-in world: more held-out cases, used only after human review.

    python -m evals.worlds.lived.round2 --per-kind 2 --budget 1              # write round2-s1.json
    python -m evals.worlds.lived.round2 --per-kind 2 --budget 1 --fake-llm   # offline dry run
    python -m evals.worlds.lived.round2 --apply decisions.json               # write round2-s1.reviewed.json

The world is not regrown. Each snapshot is rebuilt from the saved size entry plus the deterministic
plan: roster, day and thread map as saved; the transcript is the saved probes' context (its last
CONTEXT entries, all build_probes reads); thread_last_day and last_user_threads replay the plan's
events up to the snapshot's last played event, which the context's message timestamps pin down.
That replay is exact only when growth skipped no event, so a world with skips is refused.

Every round-2 probe is test-only (split forced to "test" whatever its id hash) and joins a run
only once a reviewed file lists it as accept or fixed. With no reviewed file, nothing changes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...cases import NEW, Case
from ...llm import LLMClient, ScriptedLLM
from ..model import World
from .model import LifePlan, Snapshot, day_to_datetime
from .plan import build_plan
from .probes import CONTEXT, build_probes
from .store import DEFAULT_PATH, _agent, _case

SIZES = (50, 200)
FULL_SIZE = 1000  # reuses the 200 snapshot's moment and probes, as the original does
ROUND = "r2"
FORMAT = 1
DECISIONS = ("accept", "fixed", "reject")
MESSAGE_TAGS = {"user_message": "user", "agent_message": "agent"}


def round2_path(world_path: Path, seed: int) -> Path:
    return Path(world_path).with_name(f"round2-s{seed}.json")


def reviewed_path(world_path: Path, seed: int) -> Path:
    return Path(world_path).with_name(f"round2-s{seed}.reviewed.json")


def rng_seed(seed: int, size: int) -> int:
    """Distinct from the original probes' seed * 1000 + size for every seed and size."""
    return 2_000_000 + seed * 1000 + size


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _norm(message: str) -> str:
    return " ".join(message.lower().split())


class TestCase(Case):
    """A round-2 case: always in the test split, whatever its id hashes to."""

    __test__ = False  # not a pytest class

    def __post_init__(self) -> None:
        self.split_override = "test"
        super().__post_init__()


# --- snapshot rebuild ---------------------------------------------------------------------------


def _last_event_index(plan: LifePlan, day: float, context: Sequence[Sequence[str]]) -> int:
    """Index of the snapshot's last played event: the one at `day` whose run of preceding events
    matches, one to one, the context's message entries (tag and timestamp)."""
    messages = [entry for entry in context if entry[0] in MESSAGE_TAGS]
    if not messages:
        raise ValueError("snapshot context holds no message entry to anchor the replay")
    matches = []
    for index, event in enumerate(plan.events):
        if event.day != day or index + 1 < len(messages):
            continue
        run = plan.events[index + 1 - len(messages): index + 1]
        if all(
            entry[2] == day_to_datetime(ev.day).isoformat() and MESSAGE_TAGS[entry[0]] == ev.intent.message_type
            for entry, ev in zip(messages, run)
        ):
            matches.append(index)
    if len(matches) != 1:
        raise ValueError(f"cannot pin the snapshot at day {day} to one plan event (matches: {matches})")
    return matches[0]


def rebuild_snapshot(entry: dict[str, Any], plan: LifePlan, *, skipped: int) -> Snapshot:
    """The Snapshot build_probes needs, from one saved size entry and the plan; see the module doc.

    Worlds saved with their probe state (`transcript` in the entry) rebuild from it directly; older
    worlds (life 1) replay the plan from the probes' shared context."""
    if "transcript" in entry:
        return Snapshot(
            size=entry["size"], day=entry["day"], agents=[_agent(a) for a in entry["agents"]],
            transcript=[tuple(item) for item in entry["transcript"]],
            thread_agents={t: list(ids) for t, ids in entry["thread_agents"].items()},
            thread_last_day={t: float(d) for t, d in entry["thread_last_day"].items()},
            last_user_threads=tuple(entry["last_user_threads"]),
        )
    if skipped:
        raise ValueError(f"growth skipped {skipped} events; thread_last_day cannot be replayed exactly")
    contexts = {json.dumps(case["context"]) for case in entry["cases"]}
    if len(contexts) != 1:
        raise ValueError(f"{entry['label']}: probes disagree on the context ({len(contexts)} variants)")
    transcript = [tuple(item) for item in json.loads(contexts.pop())]
    if len(transcript) < CONTEXT:
        raise ValueError(f"{entry['label']}: context has {len(transcript)} entries, fewer than {CONTEXT}")
    last = _last_event_index(plan, entry["day"], transcript)
    thread_last_day: dict[str, float] = {}
    last_user: tuple[str, ...] = ()
    for event in plan.events[: last + 1]:
        for thread_id in event.intent.thread_ids:
            thread_last_day[thread_id] = event.day
        if event.intent.message_type == "user":
            last_user = tuple(event.intent.thread_ids)
    return Snapshot(
        size=entry["size"], day=entry["day"], agents=[_agent(a) for a in entry["agents"]], transcript=transcript,
        thread_agents={t: list(ids) for t, ids in entry["thread_agents"].items()},
        thread_last_day=thread_last_day, last_user_threads=last_user,
    )


# --- generation ---------------------------------------------------------------------------------


def _labels(size: int, sizes_saved: set[int]) -> list[str]:
    labels = [f"lived-{size}"]
    if size == max(SIZES) and FULL_SIZE in sizes_saved:
        labels.append(f"lived-{FULL_SIZE}")
    return labels


def _probe_record(case: Case, size: int, labels: list[str], roster: dict[str, Any]) -> dict[str, Any]:
    def agent(agent_id: str, role: str) -> dict[str, Any]:
        spec = roster[agent_id]
        return {"id": agent_id, "name": spec["name"], "purpose": spec["purpose"], "role": role}

    gold_agents = [agent(a, "gold") for a in case.gold_targets if a != NEW]
    gold_agents += [agent(a, "acceptable") for a in case.acceptable_ids]
    return {
        "id": case.id, "status": "unreviewed", "size": size, "labels": labels, "kind": case.kind,
        "world_seed": case.world_seed, "message": case.message, "message_type": case.message_type,
        "gold_targets": case.gold_targets, "acceptable_ids": case.acceptable_ids,
        "clarify_ok": case.clarify_ok, "dispatch_optional": case.dispatch_optional,
        "gold_agents": gold_agents, "context": [list(c) for c in case.context],
    }


async def generate_round2(
    *, world_path: Path, llm: LLMClient, model: str, per_kind: int, fake: bool = False, log=print
) -> dict[str, Any]:
    """The round-2 file body; no file is written."""
    data = json.loads(Path(world_path).read_text(encoding="utf-8"))
    seed = data["seed"]
    plan = build_plan(seed)
    entries = {entry["size"]: entry for entry in data["sizes"]}
    existing = {_norm(c["message"]) for entry in data["sizes"] for c in entry["cases"]}
    existing_ids = {c["id"] for entry in data["sizes"] for c in entry["cases"]}
    seen = set(existing)
    probes: list[dict[str, Any]] = []
    duplicates: list[dict[str, str]] = []
    rosters: dict[str, list[dict[str, Any]]] = {}
    for size in SIZES:
        entry = entries[size]
        snapshot = rebuild_snapshot(entry, plan, skipped=data["meta"].get("skipped", 0))
        roster = {a["id"]: a for a in entry["agents"]}
        rosters[entry["label"]] = [
            {"id": a["id"], "name": a["name"], "purpose": a["purpose"], "threads": a.get("threads", [])}
            for a in entry["agents"]
        ]
        cases = await build_probes(snapshot, plan, llm=llm, model=model, per_kind=per_kind,
                                   rng_seed=rng_seed(seed, size))
        kept_per_kind: dict[str, int] = {}
        labels = _labels(size, set(entries))
        for case in cases:
            if _norm(case.message) in seen:
                duplicates.append({"size": str(size), "kind": case.kind, "message": case.message})
                continue
            seen.add(_norm(case.message))
            index = kept_per_kind.get(case.kind, 0)
            kept_per_kind[case.kind] = index + 1
            case = replace(case, id=f"lived-s{seed}-{size}-{ROUND}-{case.kind}-{index:02d}")
            if case.id in existing_ids:
                raise ValueError(f"round-2 id {case.id} collides with an existing case")
            probes.append(_probe_record(case, size, labels, roster))
        log(f"size {size}: {len(cases)} drafted, {sum(kept_per_kind.values())} kept")
    return {
        "format": FORMAT, "seed": seed, "round": 2,
        "world": Path(world_path).name, "world_sha256": _sha256(world_path),
        "meta": {
            "model": model, "fake_llm": fake, "per_kind": per_kind, "generated_at": datetime.now(timezone.utc).isoformat(),
            "rng_seeds": {str(size): rng_seed(seed, size) for size in SIZES}, "skipped_duplicates": duplicates,
        },
        "rosters": rosters,
        "probes": probes,
    }


# --- review -------------------------------------------------------------------------------------


def apply_decisions(round2: dict[str, Any], decisions: list[dict[str, Any]], *, source_sha256: str) -> dict[str, Any]:
    """The reviewed file body: every decided probe as a ready Case, with its decision.

    A decision is {id, decision: accept|fixed|reject, gold_targets?, acceptable_ids?, clarify_ok?, note?};
    the optional fields are only allowed (and at least one is required) for fixed. Undecided probes
    are left out, so they never run."""
    if round2.get("meta", {}).get("fake_llm"):
        raise ValueError("refusing to review a --fake-llm round-2 file")
    by_id = {p["id"]: p for p in round2["probes"]}
    rosters = {label: {a["id"] for a in agents} for label, agents in round2["rosters"].items()}
    out, seen = [], set()
    for item in decisions:
        probe_id, decision = item.get("id"), item.get("decision")
        if probe_id not in by_id:
            raise ValueError(f"unknown probe id {probe_id!r}")
        if probe_id in seen:
            raise ValueError(f"probe {probe_id} decided twice")
        seen.add(probe_id)
        if decision not in DECISIONS:
            raise ValueError(f"{probe_id}: decision must be one of {DECISIONS}, not {decision!r}")
        probe = by_id[probe_id]
        fixes = {k: item[k] for k in ("gold_targets", "acceptable_ids", "clarify_ok") if k in item}
        if decision == "fixed" and not fixes:
            raise ValueError(f"{probe_id}: a fixed decision needs gold_targets, acceptable_ids or clarify_ok")
        if decision != "fixed" and fixes:
            raise ValueError(f"{probe_id}: only a fixed decision may change gold")
        case = {f.name: probe[f.name] for f in fields(Case) if f.name in probe}
        case.update(fixes)
        known = rosters[probe["labels"][0]] | {NEW}
        bad = [a for a in [*case["gold_targets"], *case["acceptable_ids"]] if a not in known]
        if bad:
            raise ValueError(f"{probe_id}: gold names agents not in its roster: {bad}")
        if not isinstance(case["clarify_ok"], bool):
            raise ValueError(f"{probe_id}: clarify_ok must be true or false")
        _case(case)  # schema check
        out.append({"labels": probe["labels"], "decision": decision, "note": item.get("note", ""), "case": case})
    return {"format": FORMAT, "seed": round2["seed"], "round": 2, "source_sha256": source_sha256,
            "world_sha256": round2["world_sha256"], "probes": out}


def add_reviewed(
    world_path: Path, data: dict[str, Any], units: list[tuple[str, World, list[Case]]]
) -> list[tuple[str, World, list[Case]]]:
    """`units` plus the accepted and fixed round-2 probes of the reviewed file next to the world.

    With no reviewed file, `units` is returned as is. Called by store.load_lived."""
    path = reviewed_path(world_path, data["seed"])
    if not path.exists():
        return units
    reviewed = json.loads(path.read_text(encoding="utf-8"))
    if reviewed.get("format") != FORMAT or reviewed.get("seed") != data["seed"]:
        raise ValueError(f"{path}: wrong format or seed for {world_path}")
    if reviewed.get("world_sha256") != _sha256(world_path):
        raise ValueError(f"{path}: reviewed against a different {Path(world_path).name}")
    extra: dict[str, list[Case]] = {}
    for probe in reviewed["probes"]:
        if probe["decision"] not in ("accept", "fixed"):
            continue
        for label in probe["labels"]:
            case = probe["case"]
            extra.setdefault(label, []).append(TestCase(**{**case, "context": [tuple(c) for c in case["context"]]}))
    out = []
    for label, world, cases in units:
        added = extra.pop(label, [])
        clash = {c.id for c in cases} & {c.id for c in added}
        if clash:
            raise ValueError(f"{path}: ids already in {label}: {sorted(clash)}")
        out.append((label, world, cases + added))
    if extra:
        raise ValueError(f"{path}: probes for sizes the world does not have: {sorted(extra)}")
    return out


# --- CLI ----------------------------------------------------------------------------------------


def fake_writer() -> ScriptedLLM:
    """Offline writer: the required mentions plus a short tag hashed from the prompt."""

    def script(messages: list[dict[str, Any]], sample: int) -> dict[str, Any]:
        prompt = messages[0]["content"]
        line = re.search(r"Must include \(exactly as written\): (.*)", prompt)
        mentions = re.findall(r'"([^"]*)"', line.group(1)) if line else []
        tag = hashlib.sha1(f"{prompt}|{sample}".encode("utf-8")).hexdigest()[:6]
        return {"content": " ".join([*mentions, f"(fake {tag})"])}

    return ScriptedLLM(script)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--per-kind", type=int, default=2)
    parser.add_argument("--fake-llm", action="store_true", help="offline scripted writer; no API calls")
    parser.add_argument("--out", type=Path, help="default round2-s{seed}.json (.fake.json with --fake-llm)")
    parser.add_argument("--apply", type=Path, metavar="DECISIONS", help="write the reviewed file from decisions")
    args = parser.parse_args(argv)
    seed = json.loads(args.world.read_text(encoding="utf-8"))["seed"]
    source = round2_path(args.world, seed)

    if args.apply:
        out = args.out or reviewed_path(args.world, seed)
        body = apply_decisions(
            json.loads(source.read_text(encoding="utf-8")), json.loads(args.apply.read_text(encoding="utf-8")),
            source_sha256=_sha256(source),
        )
        out.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        kept = sum(p["decision"] != "reject" for p in body["probes"])
        print(f"{len(body['probes'])} decided, {kept} kept  wrote {out}")
        return 0

    if args.fake_llm:
        llm: Any = fake_writer()
        out = args.out or source.with_name(source.stem + ".fake.json")
    else:
        from ...llm import Budget, OpenRouterLLM
        from ...run import CACHE_DIR, _api_key  # lazy: run imports store, which imports this module

        key = _api_key()
        if not key:
            print("OPENROUTER_API_KEY is not set. Add it to .env at the repo root.", file=sys.stderr)
            return 2
        llm = OpenRouterLLM(api_key=key, cache_dir=CACHE_DIR, budget=Budget(args.budget), concurrency=8)
        out = args.out or source

    async def go() -> dict[str, Any]:
        try:
            return await generate_round2(world_path=args.world, llm=llm, model=args.model,
                                         per_kind=args.per_kind, fake=args.fake_llm)
        finally:
            await llm.aclose()

    body = asyncio.run(go())
    out.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    spent = getattr(getattr(llm, "budget", None), "spent_usd", 0.0)
    print(f"{len(body['probes'])} probes, {len(body['meta']['skipped_duplicates'])} duplicates skipped  "
          f"spent ${spent:.4f}  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
