import pytest

from evals.cases import NEW, Case
from evals.grading import Decision, grade


def case(kind="follow_up", gold=("a",), **kw):
    return Case(id="c", kind=kind, world_seed=1, message="m", gold_targets=list(gold), **kw)


@pytest.mark.parametrize(
    "the_case, decision, expected",
    [
        (case(), Decision(dispatched_ids=["a"]), "correct"),
        (case(), Decision(dispatched_ids=["b"]), "wrong_reuse"),
        (case(), Decision(created_names=["Email to Alice v2"]), "duplicate_create"),
        (case(), Decision(dispatched_ids=["a"], created_names=["x"]), "extra_create"),
        (case(), Decision(), "no_action"),
        (case(), Decision(clarified=True), "unnecessary_clarify"),
        (case(kind="new_task", gold=[NEW]), Decision(created_names=["Plumber"]), "correct"),
        (case(kind="new_task", gold=[NEW]), Decision(dispatched_ids=["a"]), "missed_create"),
        (case(kind="multi_task", gold=["a", NEW]), Decision(dispatched_ids=["a"], created_names=["Gift"]), "correct"),
        (case(kind="multi_task", gold=["a", NEW]), Decision(dispatched_ids=["a"]), "partial"),
        (case(kind="same_name_people", gold=[], clarify_ok=True), Decision(clarified=True), "correct"),
        (case(kind="same_name_people", gold=[], clarify_ok=True), Decision(dispatched_ids=["a"]), "should_clarify"),
        (case(kind="same_name_people", gold=[], clarify_ok=True), Decision(), "should_clarify"),
        (case(kind="same_person_new_topic", gold=[NEW], clarify_ok=True), Decision(clarified=True), "correct"),
        (case(kind="cross_agent", acceptable_ids=["b"]), Decision(dispatched_ids=["b"]), "correct"),
        (case(kind="agent_result", dispatch_optional=True), Decision(), "correct"),
        (case(kind="agent_result", dispatch_optional=True), Decision(dispatched_ids=["a"]), "correct"),
        (case(kind="agent_result", dispatch_optional=True), Decision(dispatched_ids=["b"]), "wrong_reuse"),
        (case(kind="agent_result", dispatch_optional=True), Decision(created_names=["x"]), "duplicate_create"),
    ],
)
def test_grade(the_case, decision, expected):
    assert grade(the_case, decision) == expected
