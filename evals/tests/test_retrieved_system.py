import json
from datetime import timedelta

import pytest

from evals.cases import Case
from evals.grading import Decision, grade
from evals.layers.retrieval import run_retrieval_case
from evals.layers.routing import build_messages, run_routing_case
from evals.llm import ScriptedLLM, tool_call
from evals.systems import get_system
from evals.systems.cards_from_world import cards_for
from evals.systems.full_cards import FullCardsSystem
from evals.systems.retrieved import RetrievedNoLifecycleSystem, RetrievedSystem
from evals.worlds.model import WORLD_NOW, AgentSpec, World
from server.agents.interaction_agent.agent import build_system_prompt
from server.services.agents.retrieval import RetrievalConfig
from server.services.agents.schemas import TOOL_SCHEMAS

GARY = "gary-whitman-lease-renewal"
BEATRIZ = "beatriz-trivia-night"
DENTIST = "dentist-cleaning-booking"
HOT = "grocery-order-this-week"

FILLER_TOPICS = [
    "Mesh Router Price Comparison", "Balance Bike Returns Research", "Standing Desk Shortlist",
    "Passport Renewal Checklist", "Gym Membership Cancellation", "Car Insurance Quotes",
    "Library Book Holds", "Plant Watering Reminders", "Recipe Meal Prep Plan", "Photo Backup Cleanup",
    "Concert Ticket Alerts", "Bike Tune Up Booking", "Tax Document Gathering", "Moving Boxes Order",
    "Laptop Warranty Claim", "Coffee Grinder Research", "Podcast Episode Notes", "Running Shoe Returns",
]


def spec(agent_id, name, purpose, requests=(), *, active=5.0, created=20.0, **gold):
    return AgentSpec(
        id=agent_id, name=name, purpose=purpose, entities=[], last_active_days=active, created_days=created,
        history=[(request, "done") for request in requests], **gold,
    )


def toy_world(n_filler=len(FILLER_TOPICS)) -> World:
    """A small lived-style world: a few people-centred agents among errands, one active just now."""
    agents = [
        spec(GARY, "Gary Whitman Lease Renewal", "Coordinate the lease renewal call with Gary Whitman.",
             ["Ask Gary Whitman when he is free to talk about the lease renewal"], active=3, created=30),
        spec(BEATRIZ, "Beatriz Trivia Night", "Plan trivia night with Beatriz Souza.",
             ["Email Beatriz about which pub to book for trivia"], active=2, created=25),
        spec(DENTIST, "Dentist Cleaning Booking", "Book a dentist cleaning appointment for next month.",
             active=20, created=40),
        spec(HOT, "Grocery Order This Week", "Place this week's grocery order.", active=0.01, created=0.01),
    ]
    for index, topic in enumerate(FILLER_TOPICS[:n_filler]):
        agents.append(spec(f"filler-{index}", topic, f"Handle the {topic.lower()}.", active=10 + index,
                           created=50 + index))
    return World(seed=0, agents=agents)


def case(message, gold, *, kind="follow_up", message_type="user", context=()):
    return Case(id=f"t-{kind}", kind=kind, world_seed=0, message=message, gold_targets=list(gold),
                message_type=message_type, context=list(context))


def agent_call(index=0, **arguments):
    """tool_call() for send_message_to_agent, whose `name` argument clashes with tool_call's own."""
    return {
        "id": f"call_send_message_to_agent_{index}",
        "type": "function",
        "function": {"name": "send_message_to_agent", "arguments": json.dumps(arguments)},
    }


def scripted(*steps):
    """Step i (the i-th assistant reply) makes those tool calls; past the last step, stop. Records messages."""
    seen = []

    def script(messages, sample):
        seen.append(list(messages))
        index = sum(1 for m in messages if m["role"] == "assistant")
        if index < len(steps):
            return {"content": "", "tool_calls": steps[index]}
        return {"content": ""}

    return ScriptedLLM(script), seen


def tool_results(messages):
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


async def run(system, world, c, *steps):
    llm, seen = scripted(*steps)
    decision = await run_routing_case(c, world, system, llm, model="m", sample=0)
    return decision, tool_results(seen[-1])


async def test_reuse_by_agent_id_dispatches_to_that_agent():
    world = toy_world()
    c = case("any word back from Gary on the lease?", [GARY])
    decision, results = await run(
        RetrievedSystem(), world, c,
        [tool_call("send_message_to_user", message="on it"),
         agent_call(1, agent_id=GARY, instructions="Check for Gary's reply")],
    )
    assert decision.dispatched_ids == [GARY]
    assert decision.created_names == []
    assert results[1]["status"] == "success"
    assert results[1]["result"] == {
        "status": "submitted", "agent_id": GARY, "agent_name": "Gary Whitman Lease Renewal", "new_agent_created": False,
    }
    assert grade(c, decision) == "correct"


async def test_repeat_reuse_is_recorded_once():
    world = toy_world()
    send = agent_call(agent_id=GARY, instructions="x")
    decision, _ = await run(RetrievedSystem(), world, case("gary?", [GARY]), [send], [send])
    assert decision.dispatched_ids == [GARY]


async def test_create_by_name_and_purpose():
    world = toy_world()
    c = case("can you find me a plumber for the kitchen sink", ["NEW"], kind="new_task")
    decision, results = await run(
        RetrievedSystem(), world, c,
        [agent_call(name="Kitchen Sink Plumber", purpose="Find a plumber for the kitchen sink.",
                   instructions="Find a plumber")],
    )
    assert decision.created_names == ["Kitchen Sink Plumber"]
    assert decision.dispatched_ids == []
    assert results[0]["status"] == "success" and results[0]["result"]["new_agent_created"] is True
    assert grade(c, decision) == "correct"


async def test_reusing_an_agent_created_this_turn_is_not_a_dispatch():
    world = toy_world()
    create = agent_call(name="Kitchen Sink Plumber", purpose="Find a plumber for the sink.",
                       instructions="Find a plumber")
    again = agent_call(agent_id="kitchen-sink-plumber", instructions="Also ask about price")
    c = case("plumber", ["NEW"], kind="new_task")
    decision, results = await run(RetrievedSystem(), world, c, [create], [again])
    assert decision.created_names == ["Kitchen Sink Plumber"]
    assert decision.dispatched_ids == []
    assert results[1]["status"] == "success" and results[1]["result"]["agent_id"] == "kitchen-sink-plumber"


async def test_possible_duplicate_has_no_effect_then_force_new_creates():
    world = toy_world()
    args = {"name": "Gary Whitman Lease Renewal Call", "purpose": "Set up the lease renewal call with Gary Whitman.",
            "instructions": "Email Gary about the call"}
    decision, results = await run(
        RetrievedSystem(), world, case("set up a call with gary about the lease", [GARY]),
        [agent_call(**args)],
        [agent_call(**args, force_new=True)],
    )
    duplicate, forced = results
    assert duplicate["status"] == "error"
    assert duplicate["error"]["status"] == "possible_duplicate"
    assert f'agent_id="{GARY}"' in duplicate["error"]["instructions"]
    assert f'id="{GARY}"' in duplicate["error"]["candidate"]
    assert forced["status"] == "success" and forced["result"]["new_agent_created"] is True
    assert decision.created_names == ["Gary Whitman Lease Renewal Call"]
    assert decision.dispatched_ids == []


async def test_possible_duplicate_alone_is_no_action():
    world = toy_world()
    decision, results = await run(
        RetrievedSystem(), world, case("gary lease", [GARY]),
        [agent_call(name="Gary Whitman Lease Renewal", purpose="Lease call with Gary Whitman.",
                   instructions="x")],
    )
    assert results[0]["error"]["status"] == "possible_duplicate"
    assert decision.created_names == [] and decision.dispatched_ids == []


async def test_search_then_dispatch():
    world = toy_world()
    decision, results = await run(
        RetrievedSystem(), world, case("what did the dentist say?", [DENTIST], kind="dormant_revival"),
        [tool_call("search_agents", query="dentist cleaning")],
        [agent_call(agent_id=DENTIST, instructions="Any reply from the dentist?")],
    )
    search, sent = results
    assert search["status"] == "success"
    assert search["result"].startswith("<active_agents shown=")
    assert f'id="{DENTIST}"' in search["result"]
    assert sent["status"] == "success"
    assert decision.dispatched_ids == [DENTIST]


async def test_unknown_agent_id_is_an_error_with_no_effect():
    world = toy_world()
    decision, results = await run(
        RetrievedSystem(), world, case("gary?", [GARY]),
        [agent_call(agent_id="gary-whitman", instructions="x")],
    )
    assert results[0]["status"] == "error"
    assert "Unknown agent_id 'gary-whitman'" in results[0]["error"]["error"]
    assert f'id="{GARY}"' in results[0]["error"]["nearest_agents"]
    assert decision.dispatched_ids == [] and decision.created_names == []


async def test_missing_name_or_purpose_is_an_error_with_no_effect():
    world = toy_world()
    decision, results = await run(
        RetrievedSystem(), world, case("x", ["NEW"], kind="new_task"),
        [agent_call(name="Only a name", instructions="x")],
    )
    assert results[0]["status"] == "error"
    assert decision.created_names == []


async def test_missing_instructions_uses_the_shim_error():
    world = toy_world()
    decision, results = await run(
        RetrievedSystem(), world, case("x", [GARY]), [agent_call(agent_id=GARY)],
    )
    assert results[0]["status"] == "error"
    assert "missing 1 required positional argument: 'instructions'" in results[0]["error"]["error"]
    assert decision.dispatched_ids == []


async def test_each_turn_gets_a_fresh_directory():
    world = toy_world()
    system = RetrievedSystem()
    create = {"name": "Kitchen Sink Plumber", "purpose": "Find a plumber for the sink.", "instructions": "x"}
    for _ in range(2):
        decision = Decision()
        result = system.new_turn(world, case("x", ["NEW"])).handle("send_message_to_agent", create, decision)
        assert result["status"] == "success" and decision.created_names == ["Kitchen Sink Plumber"]
    assert all(card.id != "kitchen-sink-plumber" for card in system.cards(world))


def test_roster_section_is_the_rendered_cards_and_replaces_the_roster():
    world = toy_world()
    c = case("any word back from Gary on the lease?", [GARY])
    roster = RetrievedSystem().render_roster(world, c)
    assert roster.startswith('<active_agents shown="') and f'total="{len(world.agents)}"' in roster
    assert roster.endswith("</active_agents>")
    content = build_messages(c, world, RetrievedSystem())[0]["content"]
    assert roster in content and content.count("<active_agents") == 1


def test_shown_ids_match_the_rendered_roster():
    world = toy_world()
    c = case("any word back from Gary on the lease?", [GARY])
    system = RetrievedSystem()
    roster = system.render_roster(world, c)
    shown = system.shown_ids(world, c)
    assert f'shown="{len(shown)}"' in roster
    assert [roster.index(f'id="{agent_id}"') for agent_id in shown] == sorted(
        roster.index(f'id="{agent_id}"') for agent_id in shown
    )


def test_hot_agent_leads_and_gold_is_retrieved_among_a_few():
    world = toy_world()
    c = case("any word back from Gary on the lease?", [GARY])
    shown = RetrievedSystem().shown_ids(world, c)
    assert shown[0] == HOT  # active within the hour
    assert GARY in shown and len(shown) < len(world.agents)


def test_agent_message_pins_the_reporting_agent_first():
    world = toy_world()
    message = "[SUCCESS] Dentist Cleaning Booking: the clinic replied with Tuesday 3pm."
    system = RetrievedSystem()
    as_agent = system.shown_ids(world, case(message, [DENTIST], kind="agent_result", message_type="agent"))
    as_user = system.shown_ids(world, case(message, [DENTIST], kind="agent_result"))
    assert as_agent[0] == DENTIST
    assert as_user[0] == HOT


def test_pinned_card_appears_even_when_retrieval_would_drop_it():
    world = toy_world()
    message = "[SUCCESS] Dentist Cleaning Booking: done."
    system = RetrievedSystem(cfg=RetrievalConfig(k=0, hot_max=0))
    assert system.shown_ids(world, case(message, [DENTIST], kind="agent_result", message_type="agent")) == [DENTIST]
    assert system.shown_ids(world, case(message, [DENTIST], kind="agent_result")) == []


def test_query_uses_recent_context():
    world = toy_world()
    world.by_id()[BEATRIZ].last_active_days = 20  # dormant: too stale to be shown on recency alone
    context = [("user_message", "let's sort out trivia with Beatriz", "2026-10-27T11:00:00"),
               ("assistant_message", "Sure, emailing her now.", "2026-10-27T11:00:05")]
    system = RetrievedSystem()
    assert BEATRIZ in system.shown_ids(world, case("did she reply?", [BEATRIZ], context=context))
    assert BEATRIZ not in system.shown_ids(world, case("did she reply?", [BEATRIZ]))


def test_l1_recall_on_a_toy_world():
    world = toy_world()
    cases = [
        case("any word back from Gary on the lease?", [GARY]),
        case("ask beatriz if thursday works for trivia", [BEATRIZ]),
        case("did the dentist confirm the cleaning?", [DENTIST], kind="dormant_revival"),
        case("ask baetriz about the pub", [BEATRIZ], kind="typo"),
    ]
    for c in cases:
        row = run_retrieval_case(c, world, RetrievedSystem())
        assert row["recall"] == 1.0, c.message
        assert row["shown_count"] < len(world.agents)


def test_full_cards_shows_every_card_in_roster_order():
    world = toy_world()
    c = case("any word back from Gary on the lease?", [GARY])
    system = FullCardsSystem()
    assert system.shown_ids(world, c) == [agent.id for agent in world.roster_order()]
    roster = system.render_roster(world, c)
    assert f'<active_agents shown="{len(world.agents)}" total="{len(world.agents)}">' in roster
    row = run_retrieval_case(c, world, system)
    assert row["shown_count"] == len(world.agents) and row["recall"] == 1.0


async def test_full_cards_dispatches_like_retrieved():
    world = toy_world()
    decision, _ = await run(
        FullCardsSystem(), world, case("gary?", [GARY]),
        [agent_call(agent_id=GARY, instructions="x")],
    )
    assert decision.dispatched_ids == [GARY]


def test_cards_for_reads_only_registry_fields():
    world = toy_world()
    cards = {card.id: card for card in cards_for(world)}
    gary = cards[GARY]
    assert gary.name == "Gary Whitman Lease Renewal"
    assert gary.purpose == "Coordinate the lease renewal call with Gary Whitman."
    assert gary.entities == ["gary", "whitman"]
    assert gary.last_status == "" and gary.status == "active"
    assert gary.created_at == WORLD_NOW - timedelta(days=30)
    assert gary.last_active_at == WORLD_NOW - timedelta(days=3)
    assert [card.id for card in cards_for(world)] == [agent.id for agent in world.agents]


def test_cards_for_sets_lifecycle_statuses():
    world = toy_world()
    world.by_id()[BEATRIZ].last_active_days = 7
    world.by_id()["filler-0"].last_active_days = 30
    statuses = {card.id: card.status for card in cards_for(world)}
    assert statuses[GARY] == "active" and statuses[HOT] == "active"
    assert statuses[BEATRIZ] == "dormant" and statuses[DENTIST] == "dormant"
    assert statuses["filler-0"] == "archived"
    assert {card.status for card in cards_for(world, lifecycle=False)} == {"active"}


def test_lifecycle_archived_cards_surface_only_on_a_strong_match():
    world = toy_world()
    world.by_id()[DENTIST].last_active_days = 45
    c = case("did the dentist confirm the cleaning?", [DENTIST], kind="dormant_revival")
    assert DENTIST in RetrievedSystem().shown_ids(world, c)  # archived but named: still shown
    weak = case("any update on the grocery order?", [], kind="follow_up")
    assert DENTIST not in RetrievedSystem().shown_ids(world, weak)
    assert DENTIST in RetrievedSystem(lifecycle=False).shown_ids(world, c)
    assert DENTIST in RetrievedNoLifecycleSystem().shown_ids(world, c)
    # Archived is still reachable through search_agents.
    turn = RetrievedSystem().new_turn(world, c)
    found = turn.handle("search_agents", {"query": "dentist cleaning"}, Decision())
    assert f'id="{DENTIST}"' in found["result"]


def test_cards_take_entities_from_history_requests():
    agent = spec("a", "Email Priya", "email priya about the pub", ["Ask Omar Haddad whether he can come"])
    world = World(seed=0, agents=[agent])
    assert cards_for(world)[0].entities == ["priya", "omar", "haddad"]


AGENT_GOLD = {"threads", "person", "topic", "core"}
WORLD_GOLD = {"roles", "role_people", "reserved_topic"}


class PoisonedAgent(AgentSpec):
    def __getattribute__(self, name):
        if name in AGENT_GOLD:
            raise AssertionError(f"gold field AgentSpec.{name} was read")
        return super().__getattribute__(name)


class PoisonedWorld(World):
    def __getattribute__(self, name):
        if name in WORLD_GOLD:
            raise AssertionError(f"gold field World.{name} was read")
        return super().__getattribute__(name)


def poisoned(world: World) -> World:
    agents = [
        PoisonedAgent(id=a.id, name=a.name, purpose=a.purpose, entities=a.entities, last_active_days=a.last_active_days,
                      created_days=a.created_days, person="Gold Person", topic="gold", core=True,
                      history=list(a.history), threads=["gold-thread"])
        for a in world.agents
    ]
    return PoisonedWorld(seed=world.seed, agents=agents, roles={"r": GARY}, now=world.now)


@pytest.mark.parametrize("system_cls", [RetrievedSystem, FullCardsSystem])
async def test_gold_fields_are_never_read(system_cls):
    world = poisoned(toy_world())
    cards_for(world)
    system = system_cls()
    for c in (case("any word back from Gary on the lease?", [GARY]),
              case("[SUCCESS] Dentist Cleaning Booking: done.", [DENTIST], kind="agent_result", message_type="agent")):
        system.render_roster(world, c)
        system.shown_ids(world, c)
    decision, _ = await run(
        system, world, case("gary?", [GARY]),
        [tool_call("search_agents", query="gary")],
        [agent_call(agent_id=GARY, instructions="x"),
         agent_call(1, name="Gary Whitman Lease Call", purpose="Lease call with Gary Whitman.",
                   instructions="x")],
    )
    assert decision.dispatched_ids == [GARY]


def test_systems_are_registered_with_production_prompt_and_tools():
    for name, cls in (("retrieved", RetrievedSystem), ("retrieved_nolife", RetrievedNoLifecycleSystem),
                      ("full_cards", FullCardsSystem)):
        system = get_system(name)
        assert type(system) is cls and system.name == name
        assert system.system_prompt() == build_system_prompt()
        assert system.tool_schemas() is TOOL_SCHEMAS
        assert sorted(system.tool_signatures()) == sorted(s["function"]["name"] for s in TOOL_SCHEMAS)


def test_render_roster_without_a_case():
    world = toy_world()
    assert RetrievedSystem().render_roster(world, None).startswith("<active_agents shown=")
    assert RetrievedSystem().render_roster(World(seed=0, agents=[]), None) == (
        '<active_agents shown="0" total="0">None</active_agents>'
    )
