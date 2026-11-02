import json

from evals import compare
from evals.compare import kind_pass, paired


def _row(case, outcome, size="lived-200", kind="paraphrase", sample=0, expects_reuse=True):
    return {"case_id": case, "outcome": outcome, "n_agents": size, "kind": kind, "sample": sample,
            "expects_reuse": expects_reuse, "prompt_tokens": 100, "second_looks": []}


def test_paired_difference_uses_shared_cases_only():
    base = [_row("a", "duplicate_create"), _row("b", "correct"), _row("c", "correct")]
    new = [_row("a", "correct"), _row("b", "correct"), _row("z", "correct")]
    mean, low, high, n = paired(base, new, "lived-200", "pass@1")
    assert n == 2 and mean == 0.5 and low <= mean <= high
    assert paired(base, new, "lived-200", "duplicate")[0] == -0.5


def test_kind_pass():
    rows = [_row("a", "correct"), _row("a", "duplicate_create", sample=1), _row("b", "correct", kind="typo")]
    assert kind_pass(rows, "lived-200") == {"paraphrase": 0.5, "typo": 1.0}


def test_error_rows_are_excluded_even_when_every_sample_of_a_case_errored():
    base = [_row("a", "correct"), _row("b", "correct")]
    new = [_row("a", "error"), _row("a", "error", sample=1), _row("b", "duplicate_create")]
    mean, _, _, n = paired(base, new, "lived-200", "pass@1")
    assert n == 1 and mean == -1.0  # case a has no valid NEW sample, so only b is compared
    assert kind_pass(new, "lived-200") == {"paraphrase": 0.0}


def test_duplicate_counts_reuse_cases_only():
    base = [_row("a", "correct"), _row("b", "duplicate_create", expects_reuse=False)]
    new = [_row("a", "duplicate_create"), _row("b", "correct", expects_reuse=False)]
    mean, _, _, n = paired(base, new, "lived-200", "duplicate")
    assert n == 1 and mean == 1.0


def _write(tmp_path, monkeypatch, name, rows):
    (tmp_path / name).write_text(json.dumps({"routing_rows": rows}), encoding="utf-8")
    monkeypatch.setattr(compare, "RESULTS", tmp_path)


def test_report_shows_error_counts_warns_and_prints_na(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, "base.json", [_row("a", "correct")])
    (tmp_path / "new.json").write_text(json.dumps({"routing_rows": [
        _row("a", "error"), _row("b", "correct"), _row("c", "correct", size="lived-1000")]}), encoding="utf-8")
    text = compare.report("base.json", "new.json")
    assert "error rows: base 0, new 1" in text
    assert "WARNING" in text
    lived_1000 = text.split("lived-1000")[1]
    assert "n/a" in lived_1000 and "+0.000" not in lived_1000
    assert "tokens     100 -> 100" in text.split("\nlived-200")[1].split("\nlived-1000")[0]

