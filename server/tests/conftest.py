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
