import json

from evals.model_gate import check, main


def _routing(rows):
    return {"routing_rows": rows}


def _row(outcome="correct", calls=(), cap=False, fresh=0.001, prompt=1000, cached=500):
    return {"outcome": outcome, "tool_calls": list(calls), "hit_iteration_cap": cap,
            "fresh_cost": fresh, "prompt_tokens": prompt, "cached_prompt_tokens": cached}


def _e2e(statuses):
    return {"scenario_rows": [{"status": s} for s in statuses]}


def test_clean_runs_pass():
    report = check(_routing([_row(calls=[{"name": "send_message_to_agent", "arguments": {}}])] * 50), _e2e(["ok"] * 9))
    assert report["passed"] is True
    assert report["cached_share"] == 0.5 and report["fresh_cost_per_decision"] == 0.001


def test_malformed_tool_calls_fail_the_gate():
    bad = {"name": "send_message_to_agent", "arguments": {}, "error": "invalid json: Expecting value"}
    good = {"name": "send_message_to_agent", "arguments": {}}
    rows = [_row(calls=[bad])] * 3 + [_row(calls=[good])] * 97
    report = check(_routing(rows), _e2e(["ok"]))
    assert report["malformed_rate"] == 0.03 and report["passed"] is False


def test_wait_loops_and_cap_hits_fail_the_gate():
    wait = {"name": "wait", "arguments": {"reason": "x"}}
    rows = [_row(calls=[wait] * 3)] * 6 + [_row(cap=True)] * 5 + [_row()] * 89
    report = check(_routing(rows), _e2e(["ok"]))
    assert report["wait_loop_rate"] == 0.11 and report["passed"] is False


def test_errors_count_across_both_runs():
    report = check(_routing([_row(outcome="error")] + [_row()] * 49), _e2e(["error"] + ["ok"] * 49))
    assert report["error_rate"] == 0.02 and report["passed"] is True  # exactly at the threshold passes
    report = check(_routing([_row(outcome="error")] * 2 + [_row()] * 48), _e2e(["error"] + ["ok"] * 49))
    assert report["passed"] is False


def test_cli_exit_code(tmp_path):
    routing = tmp_path / "r.json"
    e2e = tmp_path / "e.json"
    routing.write_text(json.dumps(_routing([_row()] * 10)), encoding="utf-8")
    e2e.write_text(json.dumps(_e2e(["ok"] * 3)), encoding="utf-8")
    assert main(["--routing", str(routing), "--e2e", str(e2e)]) == 0
    e2e.write_text(json.dumps(_e2e(["error"] * 3)), encoding="utf-8")
    assert main(["--routing", str(routing), "--e2e", str(e2e)]) == 1
