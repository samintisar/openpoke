import pytest

from evals.cases import NEW, ROUTING_KINDS, Case, filter_cases


def _case(case_id: str, kind: str = "follow_up", gold=None, **kw) -> Case:
    return Case(id=case_id, kind=kind, world_seed=1, message="m", gold_targets=gold or ["a"], **kw)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        _case("x", kind="nonsense")


def test_expected_ids_and_new():
    case = _case("x", kind="multi_task", gold=["agent-1", NEW])
    assert case.expected_ids == ["agent-1"]
    assert case.expects_new is True
    assert case.expects_reuse is True


def test_agent_result_never_expects_reuse():
    case = _case("x", kind="agent_result", dispatch_optional=True)
    assert case.expects_reuse is False


def test_split_is_stable_and_roughly_70_30():
    cases = [_case(f"case-{i}") for i in range(400)]
    splits = [c.split for c in cases]
    assert splits == [c.split for c in cases]
    dev_share = splits.count("dev") / len(splits)
    assert 0.6 < dev_share < 0.8


def test_filter_by_split_kind_and_suite():
    cases = [_case(f"case-{i}", kind=ROUTING_KINDS[i % 2]) for i in range(50)]
    regression = {cases[0].id, cases[1].id}

    dev = filter_cases(cases, split="dev", regression_ids=regression)
    assert dev and all(c.split == "dev" for c in dev)

    only_follow = filter_cases(cases, kinds=["follow_up"], regression_ids=regression)
    assert {c.kind for c in only_follow} == {"follow_up"}

    reg = filter_cases(cases, suite="regression", regression_ids=regression)
    assert {c.id for c in reg} == regression

    cap = filter_cases(cases, suite="capability", regression_ids=regression)
    assert not ({c.id for c in cap} & regression)
