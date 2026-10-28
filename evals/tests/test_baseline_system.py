import pytest

import server.agents.interaction_agent.agent as ia_agent
from evals.systems import get_system
from evals.systems.baseline import BaselineSystem
from evals.worlds.generator import build_world
from evals.worlds.model import AgentSpec, World


class FakeRoster:
    def __init__(self, names):
        self._names = names

    def load(self):
        pass

    def get_agents(self):
        return list(self._names)


def production_roster(monkeypatch, world: World) -> str:
    """The <active_agents> section exactly as production's prompt wrapper builds it."""
    names = [a.name for a in world.roster_order()]
    monkeypatch.setattr(ia_agent, "get_agent_roster", lambda: FakeRoster(names))
    content = ia_agent.prepare_message_with_history("x", "")[0]["content"]
    start = content.index("<active_agents>")
    end = content.index("</active_agents>") + len("</active_agents>")
    return content[start:end]


def test_baseline_matches_production_rendering(monkeypatch):
    world = build_world(1).truncated(25)
    assert BaselineSystem().render_roster(world, None) == production_roster(monkeypatch, world)


def _spec(index: int, name: str) -> AgentSpec:
    return AgentSpec(id=f"t{index}", name=name, purpose="p", entities=[], last_active_days=1, created_days=10 - index)


@pytest.mark.parametrize(
    "names",
    [
        ['Say "hi" to Bo'],
        ["R&D <budget>"],
        [""],
        ['Say "hi" to Bo', "R&D <budget>", "", "plain"],
    ],
)
def test_baseline_matches_production_on_tricky_names(monkeypatch, names):
    world = World(seed=0, agents=[_spec(i, name) for i, name in enumerate(names)])
    rendered = BaselineSystem().render_roster(world, None)
    assert rendered == production_roster(monkeypatch, world)
    if "" in names:
        assert '<agent name="agent" />' in rendered


def test_empty_roster_renders_none():
    assert BaselineSystem().render_roster(World(seed=0, agents=[]), None) == "<active_agents>\nNone\n</active_agents>"


def test_shows_every_agent_in_roster_order():
    world = build_world(1).truncated(30)
    assert BaselineSystem().shown_ids(world, None) == [a.id for a in world.roster_order()]


def test_resolve_name_is_exact_like_production():
    world = build_world(1).truncated(30)
    agent = world.agents[0]
    system = BaselineSystem()
    assert system.resolve_name(world, agent.name) == agent.id
    assert system.resolve_name(world, agent.name.upper() + "!") is None


def test_get_system():
    assert get_system("baseline").name == "baseline"
    with pytest.raises(ValueError):
        get_system("nope")
