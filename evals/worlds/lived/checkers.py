"""Three-model key checks for benchmark v2: disagreements and a 10% audit sample go to human review.

    python -m evals.worlds.lived.checkers --seed 1 --seed 2 --budget 3     # rewrites bench-v2-s{seed}.json
    python -m evals.worlds.lived.checkers --seed 1 --fake-llm              # offline plumbing check

A probe's answer key comes from how the world was built, but its message was written by an LLM that can
embellish, so a key can be wrong. Three models from other families than the tested one each pick, on their
own, where the message should go among a shuffled list of candidate helpers: one helper, NEW, or ASK.

Candidates: the gold and acceptable agents; agents whose name shares a proper-noun token (a name token
that is also capitalized mid-sentence in the purpose, e.g. a person or a city) with a gold agent's name;
and what retrieval shows for the message. Sharers and retrieval results are taken in turn, gold always
kept, capped at MAX_CARDS, shuffled with an rng seeded by (seed, probe id).

Each probe gets `"check": {"answers": {model: answer}, "agree": bool, "flag": None|"disagree"|"audit"}`.
"disagree" when any model's answer does not fit the key; otherwise one draw per probe, in id order, from
random.Random(seed) puts AUDIT_SHARE of the clean probes into "audit". Both flags go to human review.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from server.services.agents.retrieval import build_query, select

from ...cases import NEW
from ...llm import LLMClient, RunAborted, ScriptedLLM
from ...systems.cards_from_world import card_for
from ..model import World
from .bench_v2 import HERE, bench_path, world_path
from .model import day_to_datetime
from .store import _agent

# Pinned ids, none from the tested DeepSeek family.
CHECKERS = ("openai/gpt-6-luna", "google/gemini-3.1-flash-lite", "qwen/qwen3.8-flash")
AUDIT_SHARE = 0.10
MAX_CARDS = 10
CONTEXT_LINES = 6
DEFAULT_BUDGET_USD = 3.0

ASK = "ASK"
INVALID = "INVALID"
ERROR = "ERROR"  # the call itself failed; never agrees

SYSTEM = (
    "A personal assistant delegates work to helper agents. Given the recent conversation, the user's new "
    "message and the numbered helpers, decide where the message should go: one helper's number, NEW if it "
    "needs a new helper, or ASK if a careful assistant should ask the user which one is meant. Reply with "
    'JSON only: {"answer": "<number>" | "NEW" | "ASK"}.'
)


# --- worlds -------------------------------------------------------------------------------------


def load_world(seed: int, size: int, directory: Path = HERE) -> World:
    """The world as it stood at the snapshot of `size` agents in life `seed`."""
    data = json.loads(world_path(seed, directory).read_text(encoding="utf-8"))
    for entry in data["sizes"]:
        if entry["size"] == size:
            return World(seed=data["seed"], agents=[_agent(a) for a in entry["agents"]],
                         now=day_to_datetime(entry["day"]))
    raise KeyError(f"life {seed} has no {size}-agent snapshot")


# --- candidates ---------------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_SENTENCE_START = re.compile(r"(^|[.!?:;]\s+|\n\s*)$")


def proper_tokens(name: str, purpose: str) -> set[str]:
    """Name tokens that the purpose also capitalizes mid-sentence: names of people, places, companies."""
    out = set()
    for token in set(_WORD.findall(name or "")):
        if len(token) < 3 or not token[0].isupper():
            continue
        for match in re.finditer(rf"\b{re.escape(token)}\b", purpose or ""):
            if not _SENTENCE_START.search(purpose[: match.start()]):
                out.add(token)
                break
    return out


def _gold_ids(probe: dict[str, Any]) -> list[str]:
    ids = [*probe.get("gold_targets", []), *probe.get("acceptable_ids", [])]
    return list(dict.fromkeys(i for i in ids if i != NEW))


def _context_texts(probe: dict[str, Any]) -> list[str]:
    return [entry[1] for entry in probe.get("context", []) if len(entry) > 1]


def candidates(probe: dict[str, Any], world: World, rng: random.Random) -> list[dict[str, str]]:
    """Shuffled {id, name, purpose} cards: gold/acceptable always, then name-sharers and retrieval in turn."""
    by_id = world.by_id()
    gold = [by_id[i] for i in _gold_ids(probe) if i in by_id]
    tokens = set().union(*(proper_tokens(a.name, a.purpose) for a in gold)) if gold else set()
    sharers = [a.id for a in world.agents if tokens & set(_WORD.findall(a.name))]
    cards = [card_for(agent, world) for agent in world.agents]
    query = build_query(probe["message"], _context_texts(probe))
    retrieved = [card.id for card in select(query, cards, world.now)]

    chosen = list(dict.fromkeys(a.id for a in gold))
    for i in range(max(len(sharers), len(retrieved))):  # one sharer, one retrieved, in turn
        for pool in (sharers, retrieved):
            if i < len(pool) and pool[i] not in chosen and len(chosen) < MAX_CARDS:
                chosen.append(pool[i])
    rng.shuffle(chosen)
    return [{"id": i, "name": by_id[i].name, "purpose": by_id[i].purpose} for i in chosen]


# --- prompt and answers -------------------------------------------------------------------------


def prompt(probe: dict[str, Any], cards: list[dict[str, str]]) -> tuple[str, str]:
    lines = ["Recent conversation:"]
    context = probe.get("context", [])[-CONTEXT_LINES:]
    lines += [f"{entry[0]}: {entry[1]}" for entry in context] or ["(none)"]
    lines.append("")
    if probe.get("message_type") == "agent":
        lines.append("New message (an agent report from a helper, not from the user):")
    else:
        lines.append("New message from the user:")
    lines += [probe["message"], "", "Helpers:"]
    lines += [f"{n}. {card['name']} — {card['purpose']}" for n, card in enumerate(cards, 1)]
    return SYSTEM, "\n".join(lines)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_ANSWER = re.compile(r'"answer"\s*:\s*"?\s*([^",}\s]+)', re.I)


def _objects(text: str) -> list[str]:
    found = [m.group(1).strip() for m in _FENCE.finditer(text)]
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                found.append(text[start:i + 1])
    return found


def _value(raw: Any, cards: list[dict[str, str]]) -> str:
    value = str(raw).strip().strip("#").strip()
    if value.upper() in (NEW, ASK):
        return value.upper()
    if value.isdigit() and 1 <= int(value) <= len(cards):
        return cards[int(value) - 1]["id"]
    return INVALID


def parse(text: str | None, cards: list[dict[str, str]]) -> str:
    """An agent id, NEW, ASK or INVALID."""
    if not text or not text.strip():
        return INVALID
    for candidate in _objects(text):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "answer" in data:
            return _value(data["answer"], cards)
    match = _ANSWER.search(text)
    return _value(match.group(1), cards) if match else INVALID


def agrees(answer: str, probe: dict[str, Any]) -> bool:
    gold = probe.get("gold_targets", [])
    clarify = bool(probe.get("clarify_ok"))
    if answer in (INVALID, ERROR):
        return False
    if answer == ASK:
        return clarify
    if answer == NEW:
        return NEW in gold or (not gold and clarify)
    return answer in gold or answer in probe.get("acceptable_ids", [])


# --- checking -----------------------------------------------------------------------------------


def _content(response: dict[str, Any]) -> str | None:
    content = response["choices"][0]["message"].get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content


async def _ask(llm: LLMClient, model: str, system: str, user: str, cards: list[dict[str, str]]) -> tuple[str, float]:
    try:
        response = await llm.complete(model=model, system=system, messages=[{"role": "user", "content": user}],
                                      tools=[], sample=0)
    except RunAborted:
        raise
    except Exception:  # one failed call is a disagreement for review, not a crash
        return ERROR, 0.0
    meta = response.get("_meta") or {}
    cost = 0.0 if meta.get("cached") else float(meta.get("cost", 0.0))
    return parse(_content(response), cards), cost


async def check_bench(bench: dict[str, Any], *, llm: LLMClient, seed: int, world_dir: Path | None = None,
                      models: tuple[str, ...] = CHECKERS) -> dict[str, Any]:
    """A copy of `bench` with a `check` on every probe and a `meta.check` summary."""
    out = copy.deepcopy(bench)
    worlds: dict[int, World] = {}
    probes = out["probes"]

    async def one(probe: dict[str, Any]) -> float:
        cards = candidates(probe, worlds[probe["size"]], random.Random(f"{seed}:{probe['id']}"))
        system, user = prompt(probe, cards)
        results = await asyncio.gather(*(_ask(llm, m, system, user, cards) for m in models))
        answers = {m: answer for m, (answer, _) in zip(models, results)}
        agree = all(agrees(a, probe) for a in answers.values())
        probe["check"] = {"answers": answers, "agree": agree, "flag": None if agree else "disagree",
                          "candidates": [c["id"] for c in cards]}
        return sum(cost for _, cost in results)

    for probe in probes:  # worlds load once per size, before the concurrent calls
        if probe["size"] not in worlds:
            worlds[probe["size"]] = load_world(seed, probe["size"], world_dir or HERE)
    costs = await asyncio.gather(*(one(p) for p in probes))

    draws = random.Random(seed)
    for probe in sorted(probes, key=lambda p: p["id"]):
        draw = draws.random()
        if probe["check"]["agree"] and draw < AUDIT_SHARE:
            probe["check"]["flag"] = "audit"
    out.setdefault("meta", {})["check"] = {"models": list(models), "audit_share": AUDIT_SHARE, "seed": seed,
                                          "max_cards": MAX_CARDS, **counts(probes), "cost_usd": round(sum(costs), 6)}
    return out


def counts(probes: list[dict[str, Any]]) -> dict[str, int]:
    flags = [p["check"]["flag"] for p in probes]
    return {"disagree": flags.count("disagree"), "audit": flags.count("audit"), "clean": flags.count(None)}


# --- fake ---------------------------------------------------------------------------------------

_CARD_LINE = re.compile(r"^(\d+)\. (.+?) — ", re.M)


def fake_llm(bench: dict[str, Any]) -> ScriptedLLM:
    """Offline: answers the number of the first gold card shown (NEW / ASK when there is none)."""
    by_message = {p["message"]: p for p in bench["probes"]}

    def script(messages: list[dict[str, Any]], sample: int) -> dict[str, Any]:
        text = messages[-1]["content"]
        body = text.split("Helpers:")[0]
        probe = next((p for m, p in by_message.items() if f"\n{m}\n" in body), None)
        numbers = {name: n for n, name in _CARD_LINE.findall(text)}
        answer = ASK
        if probe is not None:
            names = [a["name"] for a in probe.get("gold_agents", []) if a.get("role", "gold") == "gold"]
            answer = next((numbers[n] for n in names if n in numbers), NEW if NEW in probe["gold_targets"] else ASK)
        return {"content": json.dumps({"answer": answer})}

    return ScriptedLLM(script)


# --- CLI ----------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD, help="USD cap on uncached calls")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--fake-llm", action="store_true", help="offline: answers the gold card; no API calls")
    parser.add_argument("--world-dir", type=Path, default=HERE, help="where world-s{seed}.json live")
    parser.add_argument("--bench-dir", type=Path, help="where bench-v2-s{seed}.json live (default: --world-dir)")
    args = parser.parse_args(argv)
    bench_dir = args.bench_dir or args.world_dir
    benches = {seed: json.loads(bench_path(seed, directory=bench_dir).read_text(encoding="utf-8")) for seed in args.seed}

    llm: Any = None
    if not args.fake_llm:
        from ...llm import Budget, OpenRouterLLM
        from ...run import CACHE_DIR, _api_key

        key = _api_key()
        if not key:
            print("OPENROUTER_API_KEY is not set. Add it to .env at the repo root.", file=sys.stderr)
            return 2
        llm = OpenRouterLLM(api_key=key, cache_dir=CACHE_DIR, budget=Budget(args.budget), concurrency=args.concurrency)

    async def go() -> dict[int, dict[str, Any]]:
        done = {}
        for seed, bench in benches.items():
            client = fake_llm(bench) if args.fake_llm else llm
            try:
                done[seed] = await check_bench(bench, llm=client, seed=seed, world_dir=args.world_dir)
            finally:
                if args.fake_llm:
                    await client.aclose()
        if llm is not None:
            await llm.aclose()
        return done

    try:
        checked = asyncio.run(go())
    except RunAborted as exc:
        print(f"aborted: {exc}", file=sys.stderr)
        return 1
    total = 0.0
    for seed, body in checked.items():
        path = bench_path(seed, directory=bench_dir)
        path.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        c = body["meta"]["check"]
        total += c["cost_usd"]
        print(f"life {seed}: {len(body['probes'])} probes  disagree={c['disagree']} audit={c['audit']} "
              f"clean={c['clean']}  ${c['cost_usd']:.4f}  wrote {path}")
    spent = getattr(getattr(llm, "budget", None), "spent_usd", total)
    print(f"spent ${spent:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
