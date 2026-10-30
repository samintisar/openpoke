from evals.cases import Case
from evals.grading import Decision
from evals.systems import get_system
from evals.systems.full_cards import FullCardsSystem
from evals.systems.retrieved import (
    RetrievedNoFuzzySystem,
    RetrievedNoGuardSystem,
    RetrievedNoLastRoutedSystem,
    RetrievedNoLifecycleSystem,
    RetrievedNoSearchSystem,
    RetrievedSystem,
)
from evals.worlds.model import AgentSpec, World
from server.services.agents.dispatch import SEARCH_BUDGET_EXHAUSTED


def _world():
    agent = AgentSpec(id="kwame-book", name="Kwame Book Club Dates", purpose="Ask Kwame which dates work.",
                      entities=[], last_active_days=1.0, created_days=5.0)
    return World(seed=1, agents=[agent])


def _case():
    return Case(id="c", kind="vague_reference", world_seed=1, message="what about that", gold_targets=["NEW"],
                context=[], message_type="user")


async def _run(system, n):
    turn = system.new_turn(_world(), _case())
    return [await turn.handle("search_agents", {"query": f"q{i}"}, Decision()) for i in range(n)]


async def test_fourth_search_is_refused_with_the_shared_text():
    results = await _run(RetrievedSystem(), 4)
    assert [r["status"] for r in results] == ["success"] * 3 + ["error"]
    assert results[3]["error"] == SEARCH_BUDGET_EXHAUSTED


async def test_budget_resets_with_a_new_turn():
    system = RetrievedSystem()
    await _run(system, 4)
    assert [r["status"] for r in await _run(system, 3)] == ["success"] * 3


async def test_none_is_unlimited():
    results = await _run(RetrievedSystem(search_budget=None), 8)
    assert {r["status"] for r in results} == {"success"}


async def test_custom_budget():
    results = await _run(RetrievedSystem(search_budget=1), 2)
    assert [r["status"] for r in results] == ["success", "error"]


def test_default_and_forwarding():
    assert RetrievedSystem().search_budget == 3
    for cls in (RetrievedNoFuzzySystem, RetrievedNoGuardSystem, RetrievedNoLastRoutedSystem,
                RetrievedNoLifecycleSystem, RetrievedNoSearchSystem):
        assert cls().search_budget == 3 and cls(search_budget=None).search_budget is None
    assert get_system("retrieved", {"search_budget": None}).search_budget is None


def test_full_cards_is_pinned_unlimited():
    assert FullCardsSystem().search_budget is None
