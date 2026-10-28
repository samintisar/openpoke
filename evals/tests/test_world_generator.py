import hashlib

from evals.cases import NEW, ROUTING_KINDS
from evals.worlds.generator import CROSS_TOPIC, MAX_AGENTS, NEW_PEOPLE, build_world, generate_cases

SEEDS = range(1, 6)
SIZES = (10, 200, 1000)


def test_world_is_deterministic():
    assert [a.name for a in build_world(1).agents] == [a.name for a in build_world(1).agents]
    assert [a.name for a in build_world(1).agents] != [a.name for a in build_world(2).agents]


def test_full_world_has_unique_ids_and_names():
    world = build_world(2)
    assert len(world.agents) == MAX_AGENTS
    assert len({a.id for a in world.agents}) == MAX_AGENTS
    assert len({a.name for a in world.agents}) == MAX_AGENTS


def test_smaller_worlds_are_prefixes_that_keep_every_core_agent():
    world = build_world(3)
    small, mid = world.truncated(10), world.truncated(200)
    core_ids = {a.id for a in world.agents if a.core}
    assert len(small.agents) == 10 and len(mid.agents) == 200
    assert core_ids <= {a.id for a in small.agents}
    assert {a.id for a in small.agents} <= {a.id for a in mid.agents}


def test_twins_share_a_first_name_but_not_the_main_topic():
    world = build_world(4)
    main, twin = world.role_people["main"], world.role_people["twin"]
    assert main.first == twin.first and main.last != twin.last
    main_topic = world.by_id()[world.roles["main"]].topic
    assert main_topic not in {a.topic for a in world.agents if a.person == twin.full}


def test_no_filler_agent_competes_with_a_core_person_and_topic():
    for seed in SEEDS:
        world = build_world(seed)
        core_pairs = {(a.person, a.topic) for a in world.agents if a.core and a.person}
        filler_pairs = {(a.person, a.topic) for a in world.agents if not a.core and a.person}
        assert not core_pairs & filler_pairs, seed


def test_cases_cover_every_routing_kind_with_valid_gold():
    for seed in (1, 2, 3):
        world = build_world(seed)
        cases = generate_cases(world, per_kind=4)
        assert {c.kind for c in cases} == set(ROUTING_KINDS)
        assert len({c.id for c in cases}) == len(cases)
        ids = world.by_id()
        for case in cases:
            assert all(t == NEW or t in ids for t in case.gold_targets), case.id
            assert all(t in ids for t in case.acceptable_ids), case.id


def test_create_cases_really_have_no_existing_agent():
    world = build_world(6)
    new_people = {f"{first} {last}" for first, last in NEW_PEOPLE}
    assert not any(a.person in new_people for a in world.agents)
    main = world.role_people["main"]
    reserved = world.reserved_topic[0]
    assert not any(a.person == main.full and a.topic == reserved for a in world.agents)


def test_reserved_topic_and_prefix_do_not_depend_on_max_agents():
    for seed in range(1, 6):
        full, small = build_world(seed), build_world(seed, max_agents=200)
        assert full.reserved_topic == small.reserved_topic
        assert full.roles == small.roles
        prefix = [(a.id, a.name, a.purpose, a.last_active_days) for a in full.agents[:200]]
        assert prefix == [(a.id, a.name, a.purpose, a.last_active_days) for a in small.agents]


def test_cross_agent_gold_is_the_main_persons_pricing_agent():
    for seed in range(1, 6):
        world = build_world(seed)
        agents = world.by_id()
        main, colleague = world.role_people["main"], world.role_people["colleague"]
        cross = agents[world.roles["main_cross"]]
        assert world.role_people["main_cross"] == main
        assert (cross.person, cross.topic, cross.core) == (main.full, CROSS_TOPIC[0], True)
        assert colleague.full in cross.purpose
        for case in generate_cases(world, per_kind=4):
            if case.kind == "cross_agent":
                assert case.gold_targets == [cross.id]
                assert case.acceptable_ids == [world.roles["colleague"]]


def test_filler_never_uses_a_core_topic_for_a_core_person():
    for seed in range(1, 6):
        world = build_world(seed)
        core_people = {a.person for a in world.agents if a.core and a.person}
        core_topics = {a.topic for a in world.agents if a.core and a.topic}
        assert CROSS_TOPIC[0] in core_topics
        for agent in world.agents:
            if not agent.core and agent.person in core_people:
                assert agent.topic not in core_topics, agent.name


def test_every_kind_yields_per_kind_unique_cases():
    for seed in range(1, 11):
        cases = generate_cases(build_world(seed), per_kind=4)
        for kind in ROUTING_KINDS:
            of_kind = [c for c in cases if c.kind == kind]
            assert len(of_kind) == 4, (seed, kind)
            assert len({(c.message, tuple(c.context)) for c in of_kind}) == 4, (seed, kind)
            assert len({c.message for c in of_kind}) == 4, (seed, kind)
            assert [c.id for c in of_kind] == [f"{kind}-s{seed}-{i:02d}" for i in range(4)]


def _worlds_and_cases():
    """(truncated world, cases) for every seed in SEEDS and roster size in SIZES."""
    for seed in SEEDS:
        world = build_world(seed)
        cases = generate_cases(world, per_kind=4)
        for size in SIZES:
            yield world.truncated(size), cases


def test_gold_agents_are_core_and_the_only_agent_on_their_person_and_topic():
    for world, cases in _worlds_and_cases():
        agents = world.by_id()
        for case in cases:
            if case.clarify_ok:
                continue
            for gold_id in case.expected_ids:
                gold = agents[gold_id]
                assert gold.core, case.id
                rivals = [
                    a.name for a in world.agents
                    if a.id != gold_id and a.person is not None and (a.person, a.topic) == (gold.person, gold.topic)
                ]
                assert not rivals, (case.id, len(world.agents), rivals)


def test_vague_reference_context_names_the_gold_person_and_topic():
    for world, cases in _worlds_and_cases():
        agents = world.by_id()
        for case in cases:
            if case.kind == "vague_reference":
                gold = agents[case.gold_targets[0]]
                assert gold.person in case.context[0][1] and gold.topic in case.context[0][1], case.id


def test_cross_agent_gold_is_the_main_persons_pricing_approval():
    for world, cases in _worlds_and_cases():
        agents = world.by_id()
        main = world.role_people["main"]
        for case in cases:
            if case.kind == "cross_agent":
                gold = agents[case.gold_targets[0]]
                assert (gold.person, gold.topic) == (main.full, "pricing approval"), case.id


def test_typo_cases_misspell_the_first_name():
    for seed in SEEDS:
        world = build_world(seed)
        person = world.role_people["typo"]
        for case in generate_cases(world, per_kind=4):
            if case.kind != "typo":
                continue
            words = [w.strip("?,.!'") for w in case.message.split()]
            assert person.first not in words, case.message
            first = person.first
            swapped = [w for w in words if w != first and sorted(w) == sorted(first)]
            dropped = [w for w in words if any(w == first[:k] + first[k + 1:] for k in range(len(first)))]
            near = swapped + dropped
            assert near, case.message


def test_messages_are_unique_within_each_kind():
    for seed in SEEDS:
        cases = generate_cases(build_world(seed), per_kind=4)
        for kind in ROUTING_KINDS:
            messages = [c.message for c in cases if c.kind == kind]
            assert len(messages) == len(set(messages)), (seed, kind)


# Pins the exact output of generation. Any change to the world generator or case builders
# changes it: update the constant deliberately, after checking the new cases by hand.
GOLDEN_SHA256 = "289de94bc84eaedd2fb0a837425e0d0df29e8d9f132b8781099a7702936932b2"


def test_generation_matches_the_golden_hash():
    world = build_world(1)
    cases = generate_cases(world, 4)
    text = "\n".join(a.name for a in world.agents) + "\n" + "\n".join(
        f"{c.message}|{c.gold_targets}|{c.acceptable_ids}|{c.context}|{c.clarify_ok}" for c in cases
    )
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == GOLDEN_SHA256


def test_names_and_purposes_keep_the_topics_own_casing():
    for seed in SEEDS:
        for agent in build_world(seed).agents:
            # Lowercased names are deliberate near-duplicate filler; their purposes are still checked.
            name = "" if agent.name == agent.name.lower() else agent.name
            assert "Rsvp" not in name + agent.purpose and "rsvp" not in name + agent.purpose, agent.name
            assert "Approval with" not in agent.name, agent.name
