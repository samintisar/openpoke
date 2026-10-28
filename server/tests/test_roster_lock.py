"""The roster must import and round-trip on every platform."""

import importlib

from server.services.execution.roster import AgentRoster


def test_roster_round_trip(tmp_path):
    path = tmp_path / "roster.json"
    roster = AgentRoster(path)
    roster.add_agent("Email to Alice")
    roster.add_agent("Q3 Budget")

    assert AgentRoster(path).get_agents() == ["Email to Alice", "Q3 Budget"]


def test_interaction_agent_module_imports():
    module = importlib.import_module("server.agents.interaction_agent.agent")
    assert callable(module.build_system_prompt)
