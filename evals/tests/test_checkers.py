import json
import random
import re

from evals.worlds.lived import checkers
from evals.worlds.lived.checkers import (
    AUDIT_SHARE,
    CHECKERS,
    MAX_CARDS,
    agrees,
    candidates,
    check_bench,
    parse,
    prompt,
)
from evals.worlds.model import AgentSpec, World

DAY = 10.0


def _agent(aid, name, purpose):
    return {"id": aid, "name": name, "purpose": purpose, "entities": [], "last_active_days": 1.0,
            "created_days": 3.0, "history": [[purpose, "ok"]]}


AGENTS = [
    _agent("ethan-ski-trip", "Ethan Ski Trip", "Ask Ethan whether he is still up for the ski trip."),
    _agent("ethan-tax-return", "Ethan Tax Return", "Check with Ethan about the tax return numbers."),
    _agent("seattle-flights", "Seattle Flight Search", "Find flights to Seattle next Thursday."),
]


def _world_file(tmp_path, seed=1, agents=AGENTS):
    body = {"format": 1, "seed": seed, "meta": {},
            "sizes": [{"label": "lived-10", "size": 10, "day": DAY, "agents": agents, "thread_agents": {}, "cases": []}]}
    (tmp_path / f"world-s{seed}.json").write_text(json.dumps(body), encoding="utf-8")
    return tmp_path


def _probe(pid="p0", gold=("ethan-ski-trip",), clarify=False, acceptable=(), message_type="user",
           message="Is Ethan still up for the ski trip?", kind="follow_up"):
    names = {a["id"]: a["name"] for a in AGENTS}
    gold_agents = [{"id": g, "name": names[g], "purpose": "", "role": "gold"} for g in gold if g in names]
    return {"id": pid, "kind": kind, "message": message, "message_type": message_type,
            "context": [["user", "earlier line"]], "gold_targets": list(gold), "acceptable_ids": list(acceptable),
            "clarify_ok": clarify, "dispatch_optional": message_type == "agent", "size": 10,
            "labels": ["lived-10"], "gold_agents": gold_agents, "world_seed": 1}


def _bench(probes):
    return {"meta": {"seed": 1}, "rosters": {}, "probes": probes}


def _numbered(text):
    """name -> number from the prompt's numbered card lines."""
    return {m.group(2): m.group(1) for m in re.finditer(r"^(\d+)\. (.+?) — ", text, re.M)}


class PerModel:
    """A scripted client that answers per model (ScriptedLLM's script never sees the model)."""

    def __init__(self, replies):
        self.replies = replies  # model -> fn(numbers, text) -> reply
        self.calls = 0

    async def complete(self, *, model, system, messages, tools, sample):
        self.calls += 1
        text = messages[-1]["content"]
        content = self.replies[model](_numbered(text), text)
        return {"choices": [{"message": {"role": "assistant", "content": content}}],
                "_meta": {"cost": 0.0, "cached": False}}

    async def aclose(self):
        return None


def _pick(name):
    return lambda nums, text: json.dumps({"answer": nums[name]})


def _const(value):
    return lambda nums, text: value


async def _run(tmp_path, probes, replies, seed=1):
    llm = PerModel(replies)
    out = await check_bench(_bench(probes), llm=llm, seed=seed, world_dir=_world_file(tmp_path, seed=seed))
    return {p["id"]: p["check"] for p in out["probes"]}


async def test_all_agree_with_gold(tmp_path):
    checks = await _run(tmp_path, [_probe()], {m: _pick("Ethan Ski Trip") for m in CHECKERS})
    check = checks["p0"]
    assert check["agree"] is True and check["flag"] in (None, "audit")
    assert set(check["answers"]) == set(CHECKERS) and set(check["answers"].values()) == {"ethan-ski-trip"}


async def test_one_model_picks_another_card(tmp_path):
    replies = {m: _pick("Ethan Ski Trip") for m in CHECKERS}
    replies[CHECKERS[1]] = _pick("Ethan Tax Return")
    check = (await _run(tmp_path, [_probe()], replies))["p0"]
    assert check["agree"] is False and check["flag"] == "disagree"
    assert check["answers"][CHECKERS[1]] == "ethan-tax-return"


async def test_ask_agrees_only_when_clarify_ok(tmp_path):
    replies = {m: _pick("Ethan Ski Trip") for m in CHECKERS}
    replies[CHECKERS[2]] = _const('{"answer": "ASK"}')
    checks = await _run(tmp_path, [_probe("ok", clarify=True), _probe("no", clarify=False)], replies)
    assert checks["ok"]["agree"] is True and checks["ok"]["flag"] != "disagree"
    assert checks["no"]["flag"] == "disagree"


async def test_unparseable_is_invalid_and_disagrees(tmp_path):
    replies = {m: _pick("Ethan Ski Trip") for m in CHECKERS}
    replies[CHECKERS[0]] = _const("I think it's the ski one.")
    check = (await _run(tmp_path, [_probe()], replies))["p0"]
    assert check["answers"][CHECKERS[0]] == "INVALID" and check["flag"] == "disagree"


async def test_no_gold_and_clarify_ok_accepts_new_and_ask(tmp_path):
    replies = {CHECKERS[0]: _const('{"answer": "NEW"}'), CHECKERS[1]: _const('{"answer": "ASK"}'),
               CHECKERS[2]: _const('```json\n{"answer": "new"}\n```')}
    check = (await _run(tmp_path, [_probe(gold=(), clarify=True, message="How is it going?")], replies))["p0"]
    assert check["agree"] is True and check["flag"] != "disagree"


async def test_audit_is_deterministic_and_about_ten_percent(tmp_path):
    probes = [_probe(f"p{i:03d}") for i in range(200)]
    replies = {m: _pick("Ethan Ski Trip") for m in CHECKERS}
    first = await _run(tmp_path, probes, replies, seed=7)
    second = await _run(tmp_path, probes, replies, seed=7)
    assert first == second
    audited = sum(1 for c in first.values() if c["flag"] == "audit")
    assert all(c["flag"] in (None, "audit") for c in first.values())
    assert 5 <= audited <= 40 and AUDIT_SHARE == 0.10


def test_parse_variants():
    cards = [{"id": "a"}, {"id": "b"}]
    assert parse('{"answer": "2"}', cards) == "b"
    assert parse('Sure.\n```json\n{"answer": 1}\n```', cards) == "a"
    assert parse('{"answer": "3"}', cards) == "INVALID"
    assert parse('{"answer": "0"}', cards) == "INVALID"
    assert parse('{"answer": "ask"}', cards) == "ASK"
    assert parse("", cards) == "INVALID" and parse(None, cards) == "INVALID"


def test_agrees_rules():
    p = _probe(gold=("x", "NEW"), acceptable=("y",))
    assert agrees("x", p) and agrees("y", p) and agrees("NEW", p)
    assert not agrees("ASK", p) and not agrees("z", p) and not agrees("INVALID", p)
    assert not agrees("NEW", _probe(gold=("x",)))
    assert not agrees("INVALID", _probe(gold=(), clarify=True))


def test_prompt_shape():
    p = _probe(message_type="agent", message="[SUCCESS] Ethan Ski Trip: he is in")
    p["context"] = [["user", f"line {i}", "2026-01-01T00:00:00"] for i in range(9)]
    cards = [{"id": "a", "name": "Ethan Ski Trip", "purpose": "ask Ethan"}]
    system, user = prompt(p, cards)
    assert "ASK" in system and "NEW" in system and "JSON" in system
    assert "line 2" not in user and "user: line 3" in user and "line 8" in user
    assert "agent report" in user.lower() and "1. Ethan Ski Trip — ask Ethan" in user


def _many_world(n=40):
    agents = [AgentSpec(id=f"ethan-{i}", name=f"Ethan Thing {i}", purpose=f"Ask Ethan about thing {i}.",
                        entities=[], last_active_days=1.0, created_days=2.0) for i in range(n)]
    agents.append(AgentSpec(id="gold", name="Ethan Ski Trip", purpose="Ask Ethan about the ski trip.",
                            entities=[], last_active_days=50.0, created_days=60.0))
    agents.append(AgentSpec(id="alt", name="Zed Cabin", purpose="Book the cabin for Zed.",
                            entities=[], last_active_days=50.0, created_days=60.0))
    from evals.worlds.lived.model import day_to_datetime
    return World(seed=1, agents=agents, now=day_to_datetime(DAY))


def test_candidates_keep_gold_and_cap():
    world = _many_world()
    probe = _probe(gold=("gold", "NEW"), acceptable=("alt",), message="Ethan thing 3 and thing 4 and thing 5")
    cards = candidates(probe, world, random.Random("1:p0"))
    ids = [c["id"] for c in cards]
    assert {"gold", "alt"} <= set(ids) and len(ids) <= MAX_CARDS and len(set(ids)) == len(ids)
    assert "NEW" not in ids and set(cards[0]) == {"id", "name", "purpose"}
    assert any(i.startswith("ethan-") for i in ids)  # name-sharers / retrieval distractors join
    again = candidates(probe, world, random.Random("1:p0"))
    assert again == cards  # same rng seed, same order


async def test_fake_llm_agrees_everywhere(tmp_path):
    bench = _bench([_probe("a"), _probe("b", gold=("seattle-flights",), message="Any Seattle flights?"),
                    _probe("c", gold=(), clarify=True, message="How is it going?")])
    out = await check_bench(bench, llm=checkers.fake_llm(bench), seed=1, world_dir=_world_file(tmp_path))
    assert all(p["check"]["agree"] for p in out["probes"])


def test_cli_fake_llm_rewrites_bench(tmp_path, capsys):
    _world_file(tmp_path)
    (tmp_path / "bench-v2-s1.json").write_text(json.dumps(_bench([_probe()])), encoding="utf-8")
    assert checkers.main(["--seed", "1", "--fake-llm", "--world-dir", str(tmp_path), "--bench-dir", str(tmp_path)]) == 0
    body = json.loads((tmp_path / "bench-v2-s1.json").read_text(encoding="utf-8"))
    assert body["probes"][0]["check"]["agree"] is True
    out = capsys.readouterr().out
    assert "disagree" in out and "audit" in out and "clean" in out and "spent" in out
