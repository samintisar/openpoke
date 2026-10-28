import pytest

from evals.cases import ROUTING_KINDS
from evals.worlds.handwritten import load_handwritten


def test_handwritten_set_loads_and_covers_every_kind():
    world, cases = load_handwritten()
    assert len(world.agents) >= 10
    assert {c.kind for c in cases} == set(ROUTING_KINDS)


def test_unknown_agent_id_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "agents:\n"
        "  - {id: a, name: A, purpose: p, last_active_days: 1, created_days: 2}\n"
        "cases:\n"
        "  - {id: hw-x, kind: follow_up, message: m, gold_targets: [missing]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing"):
        load_handwritten(path)
