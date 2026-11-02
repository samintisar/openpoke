"""Shared test setup: the send approval gate uses a temp store, never `server/data/approvals.db`."""

import pytest

from server.services.approvals import PendingActionStore, gate, set_approval_store


@pytest.fixture(autouse=True)
def approval_store(tmp_path, monkeypatch):
    store = PendingActionStore(tmp_path / "approvals.db")
    set_approval_store(store)
    monkeypatch.setattr(gate, "enforce", True)
    monkeypatch.setattr(gate, "decision_sink", None)
    yield store
    set_approval_store(None)


@pytest.fixture(autouse=True)
def no_live_jev(monkeypatch):
    """Production's Jev reviewer is the default second look: no test may build a real client from .env's key."""
    import server.agents.interaction_agent.second_look_gate as second_look_gate

    monkeypatch.setattr(second_look_gate, "api_key_from_env", lambda: None)
    monkeypatch.setattr(second_look_gate, "_PRODUCTION_JEV", None)
    monkeypatch.setattr(second_look_gate, "_NO_KEY_WARNED", False)
