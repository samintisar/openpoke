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


AGENT_A = "  - {id: a, name: A, purpose: p, last_active_days: 1, created_days: 2}\n"
CASE_X = "  - {id: hw-x, kind: follow_up, message: m, gold_targets: [a]}\n"


def _load(tmp_path, text: str):
    path = tmp_path / "hw.yaml"
    path.write_text(text, encoding="utf-8")
    return load_handwritten(path)


def test_duplicate_agent_id_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate agent id 'a'"):
        _load(tmp_path, "agents:\n" + AGENT_A + AGENT_A + "cases:\n" + CASE_X)


def test_duplicate_case_id_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate case id 'hw-x'"):
        _load(tmp_path, "agents:\n" + AGENT_A + "cases:\n" + CASE_X + CASE_X)


def test_missing_required_key_names_the_entry(tmp_path):
    no_purpose = "  - {id: a, name: A, last_active_days: 1, created_days: 2}\n"
    with pytest.raises(ValueError, match="agent 'a'.*'purpose'"):
        _load(tmp_path, "agents:\n" + no_purpose + "cases:\n" + CASE_X)
    no_message = "  - {id: hw-x, kind: follow_up, gold_targets: [a]}\n"
    with pytest.raises(ValueError, match="case 'hw-x'.*'message'"):
        _load(tmp_path, "agents:\n" + AGENT_A + "cases:\n" + no_message)


def test_missing_top_level_key_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="'agents'"):
        _load(tmp_path, "cases:\n" + CASE_X)
    with pytest.raises(ValueError, match="'cases'"):
        _load(tmp_path, "agents:\n" + AGENT_A)


def test_empty_gold_targets_need_clarify_ok(tmp_path):
    empty = "  - {id: hw-x, kind: follow_up, message: m, gold_targets: []}\n"
    with pytest.raises(ValueError, match="hw-x.*gold_targets"):
        _load(tmp_path, "agents:\n" + AGENT_A + "cases:\n" + empty)
    ok = "  - {id: hw-x, kind: same_name_people, message: m, gold_targets: [], clarify_ok: true}\n"
    _, cases = _load(tmp_path, "agents:\n" + AGENT_A + "cases:\n" + ok)
    assert cases[0].gold_targets == []
