import pytest

import server.agents.interaction_agent.agent as ia_agent
from evals.systems import get_system
from evals.systems.baseline import BaselineSystem
from evals.worlds.generator import build_world
from evals.worlds.model import World


class FakeRoster:
    def __init__(self, names):
        self._names = names

    def load(self):
        pass

    def get_agents(self):
        return list(self._names)


def test_baseline_matches_production_rendering(monkeypatch):
    world = build_world(1).truncated(25)
    names = [a.name for a in world.roster_order()]
    monkeypatch.setattr(ia_agent, "get_agent_roster", lambda: FakeRoster(names))
    production = f"<active_agents>\n{ia_agent._render_active_agents()}\n</active_agents>"
    assert BaselineSystem().render_roster(world, None) == production


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
