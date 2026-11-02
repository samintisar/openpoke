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


def test_v2_steps_end_with_the_frozen_full_system():
    assert charts.FULL_JEV == "20261030-082330-v2-full-dev.json" and len(charts.V2_STEPS) == 6


def test_scaling_reports_pass_tokens_and_latency_per_size_and_draws_three_panels(tmp_path, monkeypatch):
    rows = [dict(r, latency_s=2.0) for r in _rows(["correct", "duplicate_create"])]
    (tmp_path / "a.json").write_text(json.dumps({"routing_rows": rows}), encoding="utf-8")
    monkeypatch.setattr(charts, "RESULTS", tmp_path)
    data = charts.scaling({"Final (this work)": "a.json"})
    mean, low, high, tokens, latency = data["Final (this work)"]["lived-200"]
    assert (mean, tokens, latency) == (0.5, 1000, 2.0) and low <= mean <= high
    svg = charts.scaling_svg({"Final (this work)": {"lived-10": data["Final (this work)"]["lived-200"],
                                                    "lived-200": data["Final (this work)"]["lived-200"]}})
    assert svg.count("<polyline") == 3 and "latency per decision" in svg
