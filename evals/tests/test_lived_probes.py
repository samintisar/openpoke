import re
from collections import Counter

from evals.cases import NEW
from evals.llm import ScriptedLLM
from evals.worlds.lived.model import Contact, LifePlan, Snapshot, Thread
from evals.worlds.lived.plan import topic_keyword
from evals.worlds.lived.probes import VAGUE_DETAIL, build_probes, corrupt_name, typo_name
from evals.worlds.model import AgentSpec

CONTACTS = [
    Contact("Alice", "Chen", "work", "Northwind Labs"),
    Contact("Alice", "Brennan", "recruiter", "Hirewell"),
    Contact("Kevin", "Doyle", "work", "Northwind Labs"),
    Contact("Ben", "Carter", "client", "Globex"),
    Contact("Rita", "Gomez", "service", "Sparkle Home"),
    Contact("Mateo", "Reyes", "family"),
]
THREADS = [
    Thread("t000", "Alice Chen", "Q3 roadmap review", "work"),
    Thread("t001", "Alice Brennan", "offer negotiation", "recruiter"),
    Thread("t002", "Kevin Doyle", "sprint planning", "work"),
    Thread("t003", "Ben Carter", "contract renewal", "client"),
    Thread("t004", "Rita Gomez", "deep clean", "service"),
    Thread("t005", None, "flight to Lisbon", "errand"),
    Thread("t006", "Mateo Reyes", "Thanksgiving plans", "family"),
    Thread("t007", "Kevin Doyle", "budget forecast", "work"),  # never got an agent
]
PLAN = LifePlan(seed=7, persona="Jordan, a product manager", contacts=CONTACTS, threads=THREADS, events=[])
THREAD_AGENTS = {
    "t000": ["roadmap-review"], "t001": ["offer-negotiation"], "t002": ["sprint-planning", "sprint-notes"],
    "t003": ["contract-renewal"], "t004": ["deep-clean"], "t005": ["lisbon-flight"], "t006": ["thanksgiving"],
}
LAST_DAY = {"t000": 95, "t001": 98, "t002": 90, "t003": 70, "t004": 50, "t005": 99, "t006": 60}


def _snapshot(transcript_extra: str = "") -> Snapshot:
    agents = [
        AgentSpec(id=a, name=a.replace("-", " ").title(), purpose=a, entities=[], last_active_days=1, created_days=5)
        for ids in THREAD_AGENTS.values() for a in ids
    ]
    transcript = [("user_message", f"message {i}", f"2026-07-01T09:{i:02d}:00+00:00") for i in range(24)]
    transcript.append(("poke_reply", f"on it {transcript_extra}".strip(), "2026-07-01T09:30:00+00:00"))
    return Snapshot(
        size=10, day=100.0, agents=agents, transcript=transcript, thread_agents=THREAD_AGENTS,
        thread_last_day=LAST_DAY, last_user_threads=("t002",),
    )


class MentionWriter(ScriptedLLM):
    """Writes the required mentions joined by spaces (or a bare pronoun) and records every prompt."""

    def __init__(self, fail: bool = False):
        self.prompts: list[str] = []

        def script(messages, sample):
            prompt = messages[0]["content"]
            self.prompts.append(prompt)
            if fail:
                return {"content": ""}
            line = re.search(r"Must include \(exactly as written\): (.*)", prompt)
            mentions = re.findall(r'"([^"]*)"', line.group(1)) if line else []
            return {"content": " ".join(mentions) or "any update on that?"}

        super().__init__(script)


async def _probes(snapshot=None, *, per_kind=10, rng_seed=3, llm=None):
    return await build_probes(
        snapshot or _snapshot(), PLAN, llm=llm or MentionWriter(), model="m", per_kind=per_kind, rng_seed=rng_seed
    )


def _by_kind(cases):
    kinds: dict[str, list] = {}
    for case in cases:
        kinds.setdefault(case.kind, []).append(case)
    return kinds


async def test_ids_are_unique_and_numbered_per_kind():
    cases = await _probes()
    assert len({c.id for c in cases}) == len(cases)
    for kind, group in _by_kind(cases).items():
        assert [c.id for c in group] == [f"lived-s7-10-{kind}-{i:02d}" for i in range(len(group))]
    assert all(c.world_seed == 7 for c in cases)


async def test_context_is_the_last_twenty_transcript_entries():
    snapshot = _snapshot()
    cases = await _probes(snapshot)
    assert all(c.context == snapshot.transcript[-20:] for c in cases)
    assert len(cases[0].context[0]) == 3


async def test_gold_follows_the_thread_map_with_extra_agents_acceptable():
    cases = await _probes()
    sprint = [c for c in _by_kind(cases)["follow_up"] if "sprint" in c.message]
    assert sprint
    assert all(c.gold_targets == ["sprint-planning"] and c.acceptable_ids == ["sprint-notes"] for c in sprint)
    for case in cases:
        if case.kind in ("follow_up", "dormant_revival", "cross_agent", "typo", "agent_result", "vague_reference"):
            ids = case.gold_targets + case.acceptable_ids
            assert ids in THREAD_AGENTS.values(), case


async def test_recency_windows():
    kinds = _by_kind(await _probes())
    active = {"t000", "t001", "t002", "t005"}
    dormant = {"t003", "t004", "t006"}
    agent_thread = {ids[0]: t for t, ids in THREAD_AGENTS.items()}
    assert {agent_thread[c.gold_targets[0]] for c in kinds["follow_up"]} == active
    assert {agent_thread[c.gold_targets[0]] for c in kinds["dormant_revival"]} == dormant


async def test_new_task_never_matches_a_plan_thread():
    cases = _by_kind(await _probes())["new_task"]
    assert cases
    for case in cases:
        assert case.gold_targets == [NEW]
        words = case.message.lower()
        assert all(words not in t.topic.lower() and words not in topic_keyword(t.topic).lower() for t in THREADS)


async def test_same_person_new_topic_uses_an_unused_topic_of_a_backed_contact():
    cases = _by_kind(await _probes())["same_person_new_topic"]
    assert cases and all(c.gold_targets == [NEW] for c in cases)
    used = {topic_keyword(t.topic).lower() for t in THREADS}
    assert all(not any(k in c.message.lower() for k in used) for c in cases)
    # A colliding first name always carries its last name.
    assert all(("Chen" in c.message or "Brennan" in c.message) for c in cases if "Alice" in c.message)


async def test_multi_task_gold_is_both_threads_or_a_thread_plus_new():
    cases = _by_kind(await _probes())["multi_task"]
    assert len(cases) >= 2
    pair, errand = cases[0], cases[1]
    assert len(pair.gold_targets) == 2 and NEW not in pair.gold_targets
    assert errand.gold_targets[-1] == NEW and len(errand.gold_targets) == 2
    agent_thread = {a: t for t, ids in THREAD_AGENTS.items() for a in ids}
    for case in cases:
        threads = {agent_thread[a] for a in case.gold_targets + case.acceptable_ids if a != NEW}
        expected = [a for t in threads for a in THREAD_AGENTS[t]]
        news = [NEW] * case.gold_targets.count(NEW)
        assert sorted(case.gold_targets + case.acceptable_ids) == sorted(expected + news)


async def test_same_name_probe_when_no_last_name_in_context():
    cases = _by_kind(await _probes())["same_name_people"]
    assert len(cases) == 1
    case = cases[0]
    assert case.gold_targets == [] and case.clarify_ok
    assert "Alice" in case.message
    assert all(w not in case.message for w in ("Chen", "Brennan", "roadmap", "negotiation"))


async def test_same_name_skipped_when_a_last_name_is_in_context():
    cases = await _probes(_snapshot("emailed alice chen"))
    assert not _by_kind(cases).get("same_name_people")


async def test_vague_reference_from_last_user_thread_forbids_names_and_topic():
    llm = MentionWriter()
    cases = _by_kind(await _probes(llm=llm))["vague_reference"]
    first = cases[0]
    assert first.message == "any update on that?"
    assert first.gold_targets == ["sprint-planning"] and first.acceptable_ids == ["sprint-notes"]
    prompt = next(p for p in llm.prompts if VAGUE_DETAIL in p)
    assert 'Must not include: "Kevin", "Doyle", "sprint"' in prompt
    assert "Must include" not in prompt
    # (b) topic-only probes name the keyword but never the person.
    for case in cases[1:]:
        thread = next(t for t, ids in THREAD_AGENTS.items() if ids[0] == case.gold_targets[0])
        contact = PLAN.contact(PLAN.thread(thread).contact)
        assert topic_keyword(PLAN.thread(thread).topic) in case.message
        assert contact.first not in case.message and contact.last not in case.message


async def test_cross_agent_names_both_people_and_golds_thread_a():
    cases = _by_kind(await _probes())["cross_agent"]
    assert cases
    for case in cases:
        thread = next(t for t, ids in THREAD_AGENTS.items() if ids[0] == case.gold_targets[0])
        a = PLAN.contact(PLAN.thread(thread).contact)
        others = [c for c in CONTACTS if c.first != a.first and c.first in case.message]
        assert a.first in case.message and others


def test_typo_name_is_deterministic_and_differs():
    for name in ("Alice", "Kevin", "Ben", "Amy", "Anna", "Bo"):
        assert typo_name(name) == typo_name(name) != name
    assert typo_name("Alice") == "Ailce" and typo_name("Ben") == "Benn"
    assert corrupt_name("tell alice and ALICE", "Alice") == "tell ailce and AILCE"


async def test_typo_probes_corrupt_the_first_name():
    cases = _by_kind(await _probes())["typo"]
    again = _by_kind(await _probes())["typo"]
    assert cases and [c.message for c in cases] == [c.message for c in again]
    for case in cases:
        thread = next(t for t, ids in THREAD_AGENTS.items() if ids[0] == case.gold_targets[0])
        first = PLAN.contact(PLAN.thread(thread).contact).first
        assert first not in case.message and typo_name(first) in case.message


async def test_agent_result_fields():
    cases = _by_kind(await _probes())["agent_result"]
    assert cases
    names = {a.id: a.name for a in _snapshot().agents}
    for case in cases:
        assert case.message.startswith(f"[SUCCESS] {names[case.gold_targets[0]]}: ")
        assert case.message_type == "agent" and case.dispatch_optional and case.clarify_ok


async def test_per_kind_is_respected():
    counts = Counter(c.kind for c in await _probes(per_kind=1))
    assert counts and max(counts.values()) == 1
    assert len(counts) == 10


async def test_same_seed_gives_the_same_probes():
    first = await _probes(rng_seed=11)
    second = await _probes(rng_seed=11)
    assert [(c.id, c.message, c.gold_targets) for c in first] == [(c.id, c.message, c.gold_targets) for c in second]


async def test_probes_whose_writer_fails_are_skipped():
    assert await _probes(llm=MentionWriter(fail=True)) == []
