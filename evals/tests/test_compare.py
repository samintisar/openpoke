from evals.compare import kind_pass, paired


def _row(case, outcome, size="lived-200", kind="paraphrase", sample=0):
    return {"case_id": case, "outcome": outcome, "n_agents": size, "kind": kind, "sample": sample}


def test_paired_difference_uses_shared_cases_only():
    base = [_row("a", "duplicate_create"), _row("b", "correct"), _row("c", "correct")]
    new = [_row("a", "correct"), _row("b", "correct"), _row("z", "correct")]
    mean, low, high, n = paired(base, new, "lived-200", "pass@1")
    assert n == 2 and mean == 0.5 and low <= mean <= high
    assert paired(base, new, "lived-200", "duplicate")[0] == -0.5


def test_kind_pass():
    rows = [_row("a", "correct"), _row("a", "duplicate_create", sample=1), _row("b", "correct", kind="typo")]
    assert kind_pass(rows, "lived-200") == {"paraphrase": 0.5, "typo": 1.0}
