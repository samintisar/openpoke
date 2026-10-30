from evals import status_wording as sw


def test_every_status_case_gets_every_template_with_its_answer_key():
    (label, _world, cases), = sw.units()
    assert label == sw.SIZE and len(cases) == len(sw.SLOTS) * len(sw.TEMPLATES)
    first = [c for c in cases if c.id.endswith("-00-w0") or "-00-w" in c.id]
    assert first[0].message == "any news on the renewal?" and first[-1].message == "has Leo replied about the renewal?"
    assert all(c.gold_targets == first[0].gold_targets and c.context == first[0].context for c in first)
    assert any(c.message == "has anyone replied about the parking?" for c in cases)


def test_only_name_dropping_rewordings_of_shared_matters_are_ambiguous():
    row = lambda case, w: {"case_id": f"v2-s1-200-status_question-{case}-w{w}"}
    assert sw.ambiguous(row("01", 0)) and not sw.ambiguous(row("01", sw.PERSON_TEMPLATE))
    assert not sw.ambiguous(row("05", 0))
