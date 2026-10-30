from server.agents.interaction_agent.agent import STATUS_POLICY, STATUS_POLICY_MARKER, build_system_prompt


def test_policy_is_in_the_prompt_by_default_and_the_marker_never_leaks():
    on, off = build_system_prompt(), build_system_prompt(status_policy=False)
    assert STATUS_POLICY in on and STATUS_POLICY not in off
    assert STATUS_POLICY_MARKER not in on and STATUS_POLICY_MARKER not in off
    assert "Never create a new agent only to check" in on
    assert len(on.split("\n")) == len(off.split("\n")) + len(STATUS_POLICY.split("\n"))
