import asyncio

import pytest

from evals import summary_drift as sd


def test_parse_answers_reads_yes_no_and_refuses_anything_else():
    text = 'ok {"supported": "yes", "current": "No", "origin": "yes", "actionable": "yes", "reason": "stale"}'
    assert sd.parse_answers(text) == {"supported": True, "current": False, "origin": True, "actionable": True,
                                      "reason": "stale"}
    with pytest.raises(ValueError):
        sd.parse_answers('{"supported": "maybe", "current": "yes", "origin": "yes", "actionable": "yes"}')
    with pytest.raises(ValueError):
        sd.parse_answers("no json here")


def test_buckets_group_by_runs_folded():
    assert [sd.bucket(n) for n in (0, 1, 2, 3, 4, 7)] == ["1 run", "1 run", "2-3 runs", "2-3 runs", "4+ runs", "4+ runs"]


def test_judge_input_lists_runs_in_order_then_the_summary():
    text = sd.judge_input("Paul Tax", [("ask Paul", ""), ("check inbox", "receipts arrived")], "Pending: none")
    assert text.index("Run 1 instruction: ask Paul") < text.index("Run 2 report: receipts arrived") < text.index("Pending: none")
    assert "Run 1 report: (none)" in text


def test_offline_run_refuses_the_sealed_life_and_counts_every_agent():
    with pytest.raises(ValueError):
        asyncio.run(sd.run(sd.fake_llm(), "m", seed=2))
    body = asyncio.run(sd.run(sd.fake_llm(), "m"))
    assert body["real"]["all"]["judged"] == 200 and body["real"]["judge_errors"] == 0
    assert body["stress"]["agents"] == sd.STRESS_AGENTS and body["stress"]["failed_updates"] == 0
