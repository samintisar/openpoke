from evals.cases import NEW, ROUTING_KINDS
from evals.worlds.generator import MAX_AGENTS, NEW_PEOPLE, build_world, generate_cases


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
    world = build_world(5)
    core_pairs = {(a.person, a.topic) for a in world.agents if a.core and a.person}
    filler_pairs = {(a.person, a.topic) for a in world.agents if not a.core and a.person}
    assert not core_pairs & filler_pairs


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
