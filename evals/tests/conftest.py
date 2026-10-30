"""Shared eval test setup: the Jev reviewer (the default second look) never reaches the network from a test."""

import pytest


@pytest.fixture(autouse=True)
def no_live_jev(monkeypatch):
    import evals.systems.retrieved as retrieved
    import server.agents.interaction_agent.second_look_gate as second_look_gate

    monkeypatch.setattr(retrieved, "api_key_from_env", lambda: None)
    monkeypatch.setattr(second_look_gate, "api_key_from_env", lambda: None)
    monkeypatch.setattr(second_look_gate, "_PRODUCTION_JEV", None)
