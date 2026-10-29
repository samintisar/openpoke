"""Standing approvals for recurring sends, and sending an approved reply/forward/draft by its action_id."""

from datetime import datetime, timedelta, timezone

import pytest

import server.agents.execution_agent.tools.gmail as gmail_tools
import server.agents.interaction_agent.tools as ia_tools
from server.services.agents.schemas import TOOL_SCHEMAS
from server.services.approvals import gate
from server.services.execution.runner import RunContext, current_run
from server.services.triggers.service import TriggerService
from server.services.triggers.store import TriggerStore
from server.tests.test_send_approvals import (  # noqa: F401 - fixtures
    MICRO,
    create_draft,
    gmail,
    in_turn,
    log,
    reply,
    run_as,
    user_says_after_showing,
)

FUTURE = "2099-01-01T09:00:00Z"
NOW = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def triggers(tmp_path, monkeypatch):
    service = TriggerService(TriggerStore(tmp_path / "triggers.db"))
    monkeypatch.setattr(ia_tools, "get_trigger_service", lambda: service)
    return service


@pytest.fixture
def run_trigger():
    """`run_trigger("alice", 3)` makes the EA tools act for a fire of trigger 3 of alice; no id: a plain run."""
    tokens = []

    def _set(agent_id, trigger_id=None):
        origin = "trigger" if trigger_id is not None else "interaction"
        tokens.append(current_run.set(RunContext(agent_id=agent_id, origin=origin, trigger_id=trigger_id)))

    yield _set
    for token in reversed(tokens):
        current_run.reset(token)


def weekly(triggers, agent_id="alice"):
    return triggers.create_trigger(
        agent_id=agent_id, agent_name=agent_id, payload="send the weekly report", recurrence_rule="FREQ=WEEKLY",
        start_time=FUTURE,
    )


def approve_recurring(trigger_id, recipients, origin="user", at=NOW):
    return in_turn(origin, at if origin == "user" else None, ia_tools.approve_recurring_send, trigger_id, recipients)


def standing(triggers, recipients=("Bob <Bob@Example.com>",)):
    trigger = weekly(triggers)
    result = approve_recurring(trigger.id, list(recipients))
    assert result.success, result.payload
    return trigger


# -- the gate ------------------------------------------------------------------------------------------------


def test_allowed_recipient_passes_from_a_trigger_run(gmail, log, triggers, run_trigger):
    trigger = standing(triggers)
    run_trigger("alice", trigger.id)
    result = reply(to="bob@example.com")
    assert result["successful"] is True and [name for name, _ in gmail.sent] == ["GMAIL_REPLY_TO_THREAD"]
    # Every fire is covered, not just one.
    assert reply(to="BOB@example.com", body="Week two.")["successful"] is True and len(gmail.sent) == 2


def test_drafts_and_forwards_from_a_trigger_run_pass_too(gmail, log, triggers, run_trigger):
    trigger = standing(triggers)
    run_trigger("alice", trigger.id)
    draft = gmail_tools.gmail_create_draft(recipient_email="bob@example.com", subject="Weekly", body="Numbers")
    assert gmail_tools.gmail_execute_draft(draft["data"]["id"])["successful"] is True
    assert gmail_tools.gmail_forward_email("m-1", "bob@example.com")["successful"] is True
    assert [name for name, _ in gmail.sent] == ["GMAIL_SEND_DRAFT", "GMAIL_FORWARD_MESSAGE"]


@pytest.mark.parametrize(
    "extra",
    [{"to": "eve@evil.com"}, {"cc": ["eve@evil.com"]}, {"bcc": ["eve@evil.com"]}, {"extra_recipients": ["e@x.com"]}],
)
def test_another_recipient_is_blocked(gmail, log, triggers, run_trigger, extra):
    trigger = standing(triggers)
    run_trigger("alice", trigger.id)
    args = {"to": "bob@example.com", **extra}
    result = reply(**args)
    assert result["status"] == "approval_required" and gmail.sent == []


def test_same_recipient_from_a_non_trigger_run_is_blocked(gmail, log, triggers, run_trigger):
    standing(triggers)
    run_trigger("alice")
    assert reply(to="bob@example.com")["status"] == "approval_required" and gmail.sent == []


def test_other_trigger_and_other_agent_are_blocked(gmail, log, triggers, run_trigger):
    trigger = standing(triggers)
    other = weekly(triggers)
    run_trigger("alice", other.id)
    assert reply(to="bob@example.com")["status"] == "approval_required"
    run_trigger("mallory", trigger.id)
    assert reply(to="bob@example.com")["status"] == "approval_required"
    assert gmail.sent == []


def test_union_of_standing_approvals_covers_the_trigger(gmail, log, triggers, run_trigger):
    trigger = standing(triggers)
    assert approve_recurring(trigger.id, ["carol@example.com"]).success
    run_trigger("alice", trigger.id)
    assert reply(to="bob@example.com", cc=["carol@example.com"])["successful"] is True


@pytest.mark.parametrize("stop", ["pause", "complete", "delete", "last_occurrence", "clear_all"])
def test_stopping_the_trigger_revokes(gmail, log, triggers, run_trigger, stop):
    trigger = standing(triggers)
    if stop == "pause":
        triggers.update_trigger(trigger.id, agent_id="alice", status="paused")
    elif stop == "complete":
        triggers.update_trigger(trigger.id, agent_id="alice", status="completed")
    elif stop == "delete":
        assert triggers.delete_trigger(trigger.id, agent_id="alice")
    elif stop == "last_occurrence":
        triggers.mark_as_completed(trigger.id, agent_id="alice")
    else:
        triggers.clear_all()
    assert gate.get_approval_store().active_standing(trigger.id, "alice") == []
    run_trigger("alice", trigger.id)
    assert reply(to="bob@example.com")["status"] == "approval_required" and gmail.sent == []


def test_resuming_a_paused_trigger_does_not_restore_the_approval(gmail, log, triggers, run_trigger):
    trigger = standing(triggers)
    triggers.update_trigger(trigger.id, agent_id="alice", status="paused")
    triggers.update_trigger(trigger.id, agent_id="alice", status="active")
    run_trigger("alice", trigger.id)
    assert reply(to="bob@example.com")["status"] == "approval_required"


def test_another_agent_cannot_pause_or_delete_to_revoke(gmail, log, triggers):
    trigger = standing(triggers)
    assert triggers.update_trigger(trigger.id, agent_id="mallory", status="paused") is None
    assert not triggers.delete_trigger(trigger.id, agent_id="mallory")
    assert len(gate.get_approval_store().active_standing(trigger.id, "alice")) == 1


def test_ea_trigger_tool_pause_revokes(gmail, log, triggers, run_trigger, monkeypatch):
    import server.agents.execution_agent.tools.triggers as trigger_tools

    trigger = standing(triggers)
    monkeypatch.setattr(trigger_tools, "_TRIGGER_SERVICE", triggers)
    monkeypatch.setattr(trigger_tools, "get_timezone_store", lambda: type("TZ", (), {"get_timezone": lambda self: "UTC"})())
    paused = trigger_tools.build_registry("alice")["updateTrigger"](trigger_id=trigger.id, status="paused")
    assert paused["status"] == "paused"
    run_trigger("alice", trigger.id)
    assert reply(to="bob@example.com")["status"] == "approval_required"


def test_gate_off_reports_would_allow_for_standing_and_blocked_sends(gmail, log, triggers, run_trigger, monkeypatch):
    decisions = []
    monkeypatch.setattr(gate, "enforce", False)
    monkeypatch.setattr(gate, "decision_sink", decisions.append)
    trigger = standing(triggers)
    run_trigger("alice", trigger.id)
    reply(to="bob@example.com")
    reply(to="eve@evil.com")
    assert len(gmail.sent) == 2
    assert [(d.allowed, d.would_allow, d.reason, d.trigger_id, bool(d.standing_approval_ids)) for d in decisions] == [
        (True, True, "standing_approval", trigger.id, True),
        (True, False, "not_approved", trigger.id, False),
    ]


# -- the IA tool ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("origin", ["agent_result", "trigger_result", "watcher", "agent"])
def test_tool_refuses_a_non_user_turn(gmail, log, triggers, origin):
    trigger = weekly(triggers)
    result = approve_recurring(trigger.id, ["bob@example.com"], origin=origin)
    assert not result.success and "Only the user" in result.payload["error"]
    assert gate.get_approval_store().active_standing(trigger.id, "alice") == []


def test_tool_refuses_outside_any_turn(gmail, log, triggers):
    trigger = weekly(triggers)
    result = ia_tools.approve_recurring_send(trigger.id, ["bob@example.com"])
    assert not result.success and "Only the user" in result.payload["error"]


def test_tool_refuses_unknown_and_inactive_triggers(gmail, log, triggers):
    assert "Unknown trigger" in approve_recurring(999, ["bob@example.com"]).payload["error"]
    assert "Invalid trigger_id" in approve_recurring("7, 8", ["bob@example.com"]).payload["error"]
    paused = weekly(triggers)
    triggers.update_trigger(paused.id, agent_id="alice", status="paused")
    result = approve_recurring(paused.id, ["bob@example.com"])
    assert not result.success and "paused" in result.payload["error"]
    assert gate.get_approval_store().active_standing(paused.id, "alice") == []


def test_tool_needs_recipients(gmail, log, triggers):
    trigger = weekly(triggers)
    assert not approve_recurring(trigger.id, []).success
    assert not approve_recurring(trigger.id, 42).success


def test_tool_binds_to_the_triggers_agent_and_normalizes(gmail, log, triggers):
    trigger = weekly(triggers, agent_id="reports")
    result = approve_recurring(str(trigger.id), ["Bob <BOB@example.com>, carol@example.com"])
    assert result.success
    assert result.payload["agent_id"] == "reports"
    assert result.payload["recipients"] == ["bob@example.com", "carol@example.com"]
    assert "approve_recurring_send" in [s["function"]["name"] for s in TOOL_SCHEMAS]
    via_dispatch = ia_tools.handle_tool_call("approve_recurring_send", {"trigger_id": trigger.id, "recipients": []})
    assert not via_dispatch.success and "Only the user" in via_dispatch.payload["error"]


# -- sending an approved action by its id --------------------------------------------------------------------


def approve_shown(action_id):
    ia_tools.send_draft(action_id=action_id)
    assert user_says_after_showing(action_id).success


def test_reply_by_action_id_sends_the_stored_content(gmail, log, run_as):
    run_as("alice")
    held = reply(body="Thanks,  Monday works.  \n", to="Dana <Dana@Example.com>", cc=["ops@example.com"])
    approve_shown(held["action_id"])
    # The agent retypes nothing: only the id (other content arguments are ignored).
    sent = gmail_tools.gmail_reply_to_thread(action_id=held["action_id"], message_body="something else")
    assert sent["successful"] is True
    (name, args), = gmail.sent
    assert name == "GMAIL_REPLY_TO_THREAD"
    assert args == {
        "thread_id": "t-7", "recipient_email": "dana@example.com", "message_body": "Thanks,  Monday works.",
        "cc": ["ops@example.com"],
    }
    assert gate.get_action(held["action_id"]).status == "sent"
    again = gmail_tools.gmail_reply_to_thread(action_id=held["action_id"])
    assert again["status"] == "refused" and "sent" in again["error"] and len(gmail.sent) == 1


def test_forward_by_action_id_sends_the_stored_content(gmail, log, run_as):
    run_as("alice")
    held = gmail_tools.gmail_forward_email("m-3", "ops@example.com", additional_text="FYI")
    approve_shown(held["action_id"])
    assert gmail_tools.gmail_forward_email(action_id=held["action_id"])["successful"] is True
    assert gmail.sent == [
        ("GMAIL_FORWARD_MESSAGE", {"message_id": "m-3", "recipient_email": "ops@example.com", "additional_text": "FYI"})
    ]
    assert gate.get_action(held["action_id"]).status == "sent"


def test_draft_by_action_id_sends_the_draft(gmail, log, run_as):
    run_as("alice")
    action_id = create_draft()["action_id"]
    approve_shown(action_id)
    assert gmail_tools.gmail_execute_draft(action_id=action_id)["successful"] is True
    assert [name for name, _ in gmail.sent] == ["GMAIL_SEND_DRAFT"]
    assert gate.get_action(action_id).status == "sent"


def test_by_action_id_refuses_when_unapproved(gmail, log, run_as):
    run_as("alice")
    held = reply()
    result = gmail_tools.gmail_reply_to_thread(action_id=held["action_id"])
    assert result["status"] == "approval_required" and result["action_id"] == held["action_id"]
    forward = gmail_tools.gmail_forward_email("m-3", "ops@example.com")
    assert gmail_tools.gmail_forward_email(action_id=forward["action_id"])["status"] == "approval_required"
    draft_action = create_draft()["action_id"]
    assert gmail_tools.gmail_execute_draft(action_id=draft_action)["status"] == "approval_required"
    assert gmail.sent == []
    assert len(gate.get_approval_store().with_status(("pending",))) == 3  # no new rows from the id sends


def test_by_action_id_refuses_another_agent(gmail, log, run_as):
    run_as("alice")
    held = reply()
    approve_shown(held["action_id"])
    run_as("mallory")
    result = gmail_tools.gmail_reply_to_thread(action_id=held["action_id"])
    assert result["status"] == "refused" and "another agent" in result["error"]
    assert gmail.sent == [] and gate.get_action(held["action_id"]).status == "approved"


def test_by_action_id_validates_kind_object_and_id(gmail, log, run_as):
    run_as("alice")
    held = reply()
    approve_shown(held["action_id"])
    assert "not a forward" in gmail_tools.gmail_forward_email(action_id=held["action_id"])["error"]
    assert "not t-8" in gmail_tools.gmail_reply_to_thread(thread_id="t-8", action_id=held["action_id"])["error"]
    assert "unknown action_id" in gmail_tools.gmail_reply_to_thread(action_id=9999)["error"]
    assert "invalid action_id" in gmail_tools.gmail_reply_to_thread(action_id="7 or 8")["error"]
    assert "not a draft" in gmail_tools.gmail_execute_draft(action_id=held["action_id"])["error"]
    assert gmail.sent == []
    assert gmail_tools.gmail_reply_to_thread(thread_id="t-7", action_id=held["action_id"])["successful"] is True


def test_by_action_id_with_an_attachment_needs_the_same_attachment(gmail, log, run_as):
    run_as("alice")
    attachment = {"s3key": "k", "name": "report.pdf", "mimetype": "application/pdf"}
    held = reply(attachment=attachment)
    approve_shown(held["action_id"])
    missing = gmail_tools.gmail_reply_to_thread(action_id=held["action_id"])
    assert missing["status"] == "refused" and "attachment" in missing["error"] and gmail.sent == []
    sent = gmail_tools.gmail_reply_to_thread(action_id=held["action_id"], attachment=attachment)
    assert sent["successful"] is True and gmail.sent[0][1]["attachment"] == attachment


def test_by_action_id_after_expiry_refuses(gmail, log, run_as, monkeypatch):
    run_as("alice")
    held = reply()
    approve_shown(held["action_id"])
    created = gate.get_action(held["action_id"]).created_at
    monkeypatch.setattr(gate, "_now", lambda: created + timedelta(days=8))
    result = gmail_tools.gmail_reply_to_thread(action_id=held["action_id"])
    assert result["status"] == "refused" and "expired" in result["error"] and gmail.sent == []


def test_missing_arguments_without_action_id_refuse(gmail, log, run_as):
    run_as("alice")
    assert gmail_tools.gmail_reply_to_thread(thread_id="t-7")["status"] == "refused"
    assert gmail_tools.gmail_forward_email(message_id="m-3")["status"] == "refused"
    assert gmail_tools.gmail_execute_draft()["status"] == "refused"


def test_approve_send_tells_the_agent_to_send_with_the_action_id(gmail, log, run_as):
    run_as("alice")
    held = reply()
    ia_tools.send_draft(action_id=held["action_id"])
    shown_at = gate.get_action(held["action_id"]).shown_at
    result = in_turn("user", shown_at + MICRO, ia_tools.approve_send, held["action_id"])
    assert f"gmail_reply_to_thread with action_id={held['action_id']}" in result.payload["next"]


def test_trigger_instructions_name_the_standing_recipients(triggers):
    from server.services.trigger_scheduler import TriggerScheduler

    scheduler = TriggerScheduler.__new__(TriggerScheduler)  # only the formatter is used
    plain = weekly(triggers)
    assert "Standing approval" not in scheduler._format_instructions(plain, NOW)
    covered = standing(triggers)
    note = scheduler._format_instructions(covered, NOW)
    assert "Standing approval" in note and "bob@example.com" in note
