import json

from evals import charts


def _rows(outcomes, size="lived-200"):
    return [{"case_id": f"c{i}", "outcome": o, "n_agents": size, "kind": "paraphrase", "sample": 0,
             "expects_reuse": True, "prompt_tokens": 1000} for i, o in enumerate(outcomes)]


def test_v2_steps_report_prints_rates_and_paired_differences(tmp_path, monkeypatch):
    for name, outcomes in {"a.json": ["duplicate_create", "correct"], "b.json": ["correct", "correct"]}.items():
        (tmp_path / name).write_text(json.dumps({"routing_rows": _rows(outcomes)}), encoding="utf-8")
    monkeypatch.setattr(charts, "RESULTS", tmp_path)
    out = charts.v2_steps_report({"first": "a.json", "second": "b.json"}, ("lived-10", "lived-200"))
    assert "lived-10" not in out and "lived-200" in out
    assert "+0.500" in out and "-0.500" in out  # pass@1 up, duplicate down, vs previous step and vs original
    assert out.count("paired difference") == 2


def test_v2_steps_skips_nothing_when_full_jev_unset():
    assert charts.FULL_JEV is None and len(charts.V2_STEPS) == 6
