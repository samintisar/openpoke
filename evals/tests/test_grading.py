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
        # over-creation: more distinct new agents than the case needs
        (case(kind="new_task", gold=[NEW]), Decision(created_names=["Plumber", "Plumber 2"]), "extra_create"),
        (case(kind="new_task", gold=[NEW]), Decision(created_names=["Plumber", "Plumber"]), "correct"),
        (
            case(kind="multi_task", gold=["a", NEW]),
            Decision(dispatched_ids=["a"], created_names=["Gift", "Gift ideas"]),
            "extra_create",
        ),
        (
            case(kind="multi_task", gold=["a", NEW]),
            Decision(dispatched_ids=["a"], created_names=["Gift", "Gift"]),
            "correct",
        ),
        (case(), Decision(dispatched_ids=["a"], created_names=["x", "y"]), "extra_create"),
        (case(kind="new_task", gold=[NEW]), Decision(dispatched_ids=["b"], created_names=["P", "Q"]), "missed_create"),
    ],
)
def test_grade(the_case, decision, expected):
    assert grade(the_case, decision) == expected


@pytest.mark.parametrize(
    "the_case, possible",
    [
        (case(), True),
        (case(kind="agent_result", dispatch_optional=True), True),
        (case(kind="new_task", gold=[NEW]), False),
        (case(kind="multi_task", gold=["a", NEW]), False),
        (case(kind="same_name_people", gold=[], clarify_ok=True), False),
    ],
)
def test_duplicate_possible_matches_grade(the_case, possible):
    from evals.grading import duplicate_possible

    assert duplicate_possible(the_case) is possible
    reachable = grade(the_case, Decision(created_names=["fresh"])) == "duplicate_create"
    assert reachable is possible
