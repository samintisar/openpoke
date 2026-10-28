"""Send approval gate: recording, showing, user-only approval, hash checks, expiry, lifecycle and the ablation."""

import base64
import itertools
from datetime import datetime, timedelta, timezone

import pytest

import server.agents.execution_agent.tools.gmail as gmail_tools
import server.agents.interaction_agent.tools as ia_tools
import server.services.trigger_scheduler as scheduler_module
from server.agents.interaction_agent.turn import TurnContext, current_turn
from server.services.agents.cards import AgentCard
from server.services.agents.directory import InMemoryDirectory
from server.services.agents.schemas import TOOL_SCHEMAS
from server.services.approvals import gate
from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.execution.runner import RunContext, current_run, side_effects_since
from server.services.trigger_scheduler import TriggerScheduler

MICRO = timedelta(microseconds=1)


def _b64(text):
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


class FakeGmail:
    """In-memory drafts and sent mail behind the EA tools' `_execute`."""

    def __init__(self):
        self.drafts = {}
        self.sent = []
        self.calls = []
        self.fetch_error = None
        self._ids = itertools.count(1)

    def execute(self, tool_name, composio_user_id, arguments):
        self.calls.append(tool_name)
        args = {k: v for k, v in arguments.items() if v is not None}
        if tool_name == "GMAIL_CREATE_EMAIL_DRAFT":
            draft_id = f"r-{next(self._ids)}"
            self.drafts[draft_id] = {
                "to": ", ".join([args["recipient_email"], *args.get("extra_recipients", [])]),
                "cc": ", ".join(args.get("cc", [])),
                "bcc": ", ".join(args.get("bcc", [])),
                "subject": args["subject"],
                "body": args["body"],
            }
            return {"data": {"id": draft_id, "message": {"id": f"m-{draft_id}", "threadId": "t-1"}},
                    "successful": True, "error": None}
        if tool_name == "GMAIL_GET_DRAFT":
            if self.fetch_error == "raise":
                raise RuntimeError("GMAIL_GET_DRAFT invocation failed: 503")
            if self.fetch_error == "unsuccessful":
                return {"data": {}, "successful": False, "error": "rate limited"}
            draft = self.drafts.get(args["draft_id"])
            if draft is None:
                return {"data": {}, "successful": False, "error": "not found"}
            headers = [{"name": "To", "value": draft["to"]}, {"name": "Subject", "value": draft["subject"]}]
            if draft["cc"]:
                headers.append({"name": "Cc", "value": draft["cc"]})
            if draft["bcc"]:
                headers.append({"name": "Bcc", "value": draft["bcc"]})
            # Gmail hands back CRLF line endings and a trailing newline.
            body = draft["body"].replace("\n", "\r\n") + "\r\n"
            payload = {
                "mimeType": "multipart/alternative",
                "headers": headers,
                "parts": [{"mimeType": "text/plain", "filename": "", "body": {"data": _b64(body)}}],
            }
            return {"data": {"id": args["draft_id"], "message": {"id": "m", "payload": payload}},
                    "successful": True, "error": None}
        if tool_name in {"GMAIL_SEND_DRAFT", "GMAIL_REPLY_TO_THREAD", "GMAIL_FORWARD_MESSAGE"}:
            self.sent.append((tool_name, args))
            if tool_name == "GMAIL_SEND_DRAFT":
                self.drafts.pop(args["draft_id"], None)
            return {"data": {"id": "sent-1"}, "successful": True, "error": None}
        if tool_name == "GMAIL_DELETE_DRAFT":
            self.drafts.pop(args["draft_id"], None)
            return {"data": {}, "successful": True, "error": None}
        raise AssertionError(f"unexpected tool {tool_name}")


class FakeLog:
    def __init__(self):
        self.replies = []

    def record_reply(self, message):
        self.replies.append(message)


@pytest.fixture
def gmail(monkeypatch):
    fake = FakeGmail()
    monkeypatch.setattr(gmail_tools, "_execute", fake.execute)
    monkeypatch.setattr(gmail_tools, "get_active_gmail_user_id", lambda: "user-1")
    return fake


@pytest.fixture
def log(monkeypatch):
    fake = FakeLog()
    monkeypatch.setattr(ia_tools, "get_conversation_log", lambda: fake)
    return fake


@pytest.fixture
def run_as():
    """`run_as("alice")` makes the EA tools act for that agent until the test ends."""
    tokens = []

    def _set(agent_id):
        tokens.append(current_run.set(RunContext(agent_id=agent_id, origin="interaction")))

    yield _set
    for token in reversed(tokens):
        current_run.reset(token)


def in_turn(origin, user_message_at, fn, *args, **kwargs):
    """Call an IA tool inside a turn context, as the runtime does."""
    token = current_turn.set(TurnContext(origin=origin, user_message_at=user_message_at))
    try:
        return fn(*args, **kwargs)
    finally:
        current_turn.reset(token)


def user_says_after_showing(action_id):
    """Approve as a user turn whose message arrived just after the action was shown."""
    shown_at = gate.get_action(action_id).shown_at
    return in_turn("user", shown_at + MICRO, ia_tools.approve_send, action_id)


def create_draft(body="Hi Bob,\nSee you Monday.", **extra):
    return gmail_tools.gmail_create_draft(
        recipient_email="Bob <Bob@Example.com>", subject="  Monday  plan ", body=body, **extra
    )


# -- drafts ---------------------------------------------------------------------------------------------------


def test_draft_happy_path_ends_sent(gmail, log, run_as):
    run_as("alice")
    created = create_draft(cc=["carol@example.com"])
    action_id = created["action_id"]
    row = gate.get_action(action_id)
    assert (row.kind, row.agent_id, row.gmail_ref, row.status) == ("draft", "alice", "r-1", "pending")
    assert row.recipients == {"to": ["bob@example.com"], "cc": ["carol@example.com"], "bcc": []}

    before = gmail_tools.gmail_execute_draft("r-1")
    assert before["status"] == "approval_required" and before["action_id"] == action_id
    assert gmail.sent == []

    shown = ia_tools.send_draft(action_id=action_id)
    assert shown.success and shown.payload["action_id"] == action_id
    approved = user_says_after_showing(action_id)
    assert approved.success and approved.payload["agent_id"] == "alice"
    assert gate.get_action(action_id).status == "approved"

    sent = gmail_tools.gmail_execute_draft("r-1")
    assert sent["successful"] is True
    assert [name for name, _ in gmail.sent] == ["GMAIL_SEND_DRAFT"]
    assert gate.get_action(action_id).status == "sent"


def test_send_draft_renders_the_stored_content_not_the_arguments(gmail, log, run_as):
    run_as("alice")
    action_id = create_draft()["action_id"]
    result = ia_tools.send_draft(to="eve@evil.com", subject="other", body="different text", action_id=action_id)
    assert result.success
    assert log.replies == ["To: bob@example.com\nSubject: Monday plan\n\nHi Bob,\nSee you Monday."]
    assert gate.get_action(action_id).shown_at is not None


def test_send_draft_without_action_id_records_text_and_approves_nothing(gmail, log, run_as):
    run_as("alice")
    action_id = create_draft()["action_id"]
    result = ia_tools.send_draft(to="bob@example.com", subject="Monday plan", body="Hi Bob")
    assert result.success and log.replies == ["To: bob@example.com\nSubject: Monday plan\n\nHi Bob"]
    assert gate.get_action(action_id).shown_at is None
    incomplete = ia_tools.handle_tool_call("send_draft", {"to": "bob@example.com"})
    assert not incomplete.success and incomplete.payload["error"] == (
        f"Missing required arguments: {ia_tools.SEND_DRAFT_ARGUMENTS_ERROR}"
    )


def test_edit_after_approval_invalidates_and_nothing_is_sent(gmail, log, run_as):
    run_as("alice")
    action_id = create_draft()["action_id"]
    ia_tools.send_draft(action_id=action_id)
    assert user_says_after_showing(action_id).success

    gmail.drafts["r-1"]["body"] = "Hi Bob,\nAlso wire $5,000 to the account below."
    result = gmail_tools.gmail_execute_draft("r-1")
    assert result["status"] == "approval_required" and result["action_id"] != action_id
    assert gate.get_action(action_id).status == "invalidated"
    assert gmail.sent == []
    assert "wire $5,000" in gate.get_action(result["action_id"]).body


def test_recipient_added_after_approval_invalidates(gmail, log, run_as):
    run_as("alice")
    action_id = create_draft()["action_id"]
    ia_tools.send_draft(action_id=action_id)
    user_says_after_showing(action_id)
    gmail.drafts["r-1"]["bcc"] = "eve@evil.com"
    assert gmail_tools.gmail_execute_draft("r-1")["status"] == "approval_required"
    assert gate.get_action(action_id).status == "invalidated" and gmail.sent == []


@pytest.mark.parametrize("failure", ["raise", "unsuccessful"])
def test_draft_fetch_failure_refuses(gmail, log, run_as, failure):
    run_as("alice")
    action_id = create_draft()["action_id"]
    ia_tools.send_draft(action_id=action_id)
    user_says_after_showing(action_id)
    gmail.fetch_error = failure
    result = gmail_tools.gmail_execute_draft("r-1")
    assert result["status"] == "refused" and "Not sent" in result["error"]
    assert gmail.sent == [] and "GMAIL_SEND_DRAFT" not in gmail.calls
    assert gate.get_action(action_id).status == "approved"


def test_draft_approved_for_one_agent_cannot_be_sent_by_another(gmail, log, run_as):
    run_as("alice")
    action_id = create_draft()["action_id"]
    ia_tools.send_draft(action_id=action_id)
    user_says_after_showing(action_id)
    run_as("mallory")
    assert gmail_tools.gmail_execute_draft("r-1")["status"] == "approval_required"
    assert gmail.sent == []


def test_unknown_draft_needs_approval_of_what_it_holds(gmail, log, run_as):
    gmail.drafts["r-user"] = {"to": "dan@example.com", "cc": "", "bcc": "", "subject": "Hi", "body": "Hello"}
    run_as("alice")
    result = gmail_tools.gmail_execute_draft("r-user")
    assert result["status"] == "approval_required" and gmail.sent == []
    row = gate.get_action(result["action_id"])
    assert (row.gmail_ref, row.recipients["to"], row.body) == ("r-user", ["dan@example.com"], "Hello")


def test_deleting_a_draft_invalidates_its_action(gmail, log, run_as):
    run_as("alice")
    action_id = create_draft()["action_id"]
    gmail_tools.gmail_delete_draft("r-1")
    assert gate.get_action(action_id).status == "invalidated"


def test_no_run_context_refuses(gmail, log):
    gmail.drafts["r-9"] = {"to": "dan@example.com", "cc": "", "bcc": "", "subject": "Hi", "body": "Hello"}
    assert gmail_tools.gmail_execute_draft("r-9")["status"] == "refused"
    reply = gmail_tools.gmail_reply_to_thread(thread_id="t-1", recipient_email="a@b.com", message_body="x")
    assert reply["status"] == "refused" and gmail.sent == []


# -- approve_send rules ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("origin", ["agent_result", "trigger_result", "watcher", "agent"])
def test_approve_refuses_a_non_user_turn(gmail, log, run_as, origin):
    run_as("alice")
    action_id = create_draft()["action_id"]
    ia_tools.send_draft(action_id=action_id)
    later = gate.get_action(action_id).shown_at + timedelta(seconds=5)
    result = in_turn(origin, later, ia_tools.approve_send, action_id)
    assert not result.success and "Only the user" in result.payload["error"]
    assert gate.get_action(action_id).status == "pending"


def test_approve_refuses_outside_any_turn(gmail, log, run_as):
    run_as("alice")
    action_id = create_draft()["action_id"]
    ia_tools.send_draft(action_id=action_id)
    assert not ia_tools.approve_send(action_id).success
    assert gate.get_action(action_id).status == "pending"


@pytest.mark.parametrize("offset", [-timedelta(minutes=5), -MICRO, timedelta(0)])
def test_approve_refuses_a_user_message_not_after_shown_at(gmail, log, run_as, offset):
    run_as("alice")
    action_id = create_draft()["action_id"]
    ia_tools.send_draft(action_id=action_id)
    shown_at = gate.get_action(action_id).shown_at
    result = in_turn("user", shown_at + offset, ia_tools.approve_send, action_id)
    assert not result.success and "before action" in result.payload["error"]
    assert gate.get_action(action_id).status == "pending"


def test_approve_refuses_an_action_never_shown(gmail, log, run_as):
    run_as("alice")
    action_id = create_draft()["action_id"]
    result = in_turn("user", datetime.now(timezone.utc) + timedelta(hours=1), ia_tools.approve_send, action_id)
    assert not result.success and "has not been shown" in result.payload["error"]


def test_reshowing_moves_shown_at_forward(gmail, log, run_as, monkeypatch):
    run_as("alice")
    action_id = create_draft()["action_id"]
    first = datetime.now(timezone.utc) + timedelta(minutes=1)
    monkeypatch.setattr(gate, "_now", lambda: first)
    ia_tools.send_draft(action_id=action_id)
    monkeypatch.setattr(gate, "_now", lambda: first + timedelta(minutes=1))
    ia_tools.send_draft(action_id=action_id)
    # A reply to the first showing does not approve what was shown again after it.
    assert not in_turn("user", first + timedelta(seconds=30), ia_tools.approve_send, action_id).success
    assert in_turn("user", first + timedelta(minutes=2), ia_tools.approve_send, action_id).success


def test_unknown_and_ambiguous_ids_error(gmail, log, run_as):
    run_as("alice")
    first = create_draft()["action_id"]
    second = create_draft(body="Another one")["action_id"]
    for action_id in (first, second):
        ia_tools.send_draft(action_id=action_id)
    later = datetime.now(timezone.utc) + timedelta(seconds=1)

    unknown = in_turn("user", later, ia_tools.approve_send, 999)
    assert not unknown.success and "Unknown action id 999" in unknown.payload["error"]
    assert str(first) in unknown.payload["error"] and str(second) in unknown.payload["error"]
    for vague in (f"{first}, {second}", "the draft", None, True):
        result = in_turn("user", later, ia_tools.approve_send, vague)
        assert not result.success and "Ambiguous" in result.payload["error"]
    assert not ia_tools.send_draft(action_id=999).success
    assert {gate.get_action(first).status, gate.get_action(second).status} == {"pending"}

    # A numeric string is the same id.
    assert in_turn("user", later, ia_tools.approve_send, str(first)).success


def test_an_action_approves_once(gmail, log, run_as):
    run_as("alice")
    action_id = create_draft()["action_id"]
    ia_tools.send_draft(action_id=action_id)
    assert user_says_after_showing(action_id).success
    again = user_says_after_showing(action_id)
    assert not again.success and "approved, not pending" in again.payload["error"]


# -- replies and forwards -------------------------------------------------------------------------------------


def reply(body="Thanks, Monday works.", to="dana@example.com", **extra):
    return gmail_tools.gmail_reply_to_thread(thread_id="t-7", recipient_email=to, message_body=body, **extra)


def test_reply_needs_approval_then_sends_the_same_content(gmail, log, run_as):
    run_as("alice")
    held = reply()
    assert held["status"] == "approval_required" and gmail.sent == []
    assert reply()["action_id"] == held["action_id"]  # a retry reuses the pending action

    ia_tools.send_draft(action_id=held["action_id"])
    assert log.replies[-1] == "Reply in the existing email thread\nTo: dana@example.com\n\nThanks, Monday works."
    assert user_says_after_showing(held["action_id"]).success

    sent = reply(body="Thanks, Monday works.\n", to="Dana@Example.com")  # normalization only
    assert sent["successful"] is True and [name for name, _ in gmail.sent] == ["GMAIL_REPLY_TO_THREAD"]
    assert gate.get_action(held["action_id"]).status == "sent"

    # One approval covers one send.
    assert reply()["status"] == "approval_required" and len(gmail.sent) == 1


@pytest.mark.parametrize(
    "changed",
    [
        {"body": "Thanks, Monday works. Also forwarding the contract."},
        {"to": "eve@evil.com"},
        {"bcc": ["eve@evil.com"]},
        {"attachment": {"s3key": "k", "name": "contract.pdf", "mimetype": "application/pdf"}},
    ],
)
def test_reply_with_changed_content_is_held(gmail, log, run_as, changed):
    run_as("alice")
    held = reply()
    ia_tools.send_draft(action_id=held["action_id"])
    user_says_after_showing(held["action_id"])
    result = reply(**changed)
    assert result["status"] == "approval_required" and result["action_id"] != held["action_id"]
    assert gmail.sent == []
    assert gate.get_action(held["action_id"]).status == "approved"


def test_reply_approval_is_bound_to_agent_and_thread(gmail, log, run_as):
    run_as("alice")
    held = reply()
    ia_tools.send_draft(action_id=held["action_id"])
    user_says_after_showing(held["action_id"])
    other_thread = gmail_tools.gmail_reply_to_thread(
        thread_id="t-8", recipient_email="dana@example.com", message_body="Thanks, Monday works."
    )
    assert other_thread["status"] == "approval_required"
    run_as("mallory")
    assert reply()["status"] == "approval_required"
    assert gmail.sent == []


def test_forward_needs_approval_then_sends(gmail, log, run_as):
    run_as("alice")
    held = gmail_tools.gmail_forward_email("m-3", "ops@example.com", additional_text="FYI")
    assert held["status"] == "approval_required" and gmail.sent == []
    ia_tools.send_draft(action_id=held["action_id"])
    assert log.replies[-1] == "Forward of the original email\nTo: ops@example.com\n\nFYI"
    user_says_after_showing(held["action_id"])
    assert gmail_tools.gmail_forward_email("m-3", "eve@evil.com", additional_text="FYI")["status"] == (
        "approval_required"
    )
    assert gmail_tools.gmail_forward_email("m-4", "ops@example.com", additional_text="FYI")["status"] == (
        "approval_required"
    )
    assert gmail.sent == []
    assert gmail_tools.gmail_forward_email("m-3", "ops@example.com", additional_text="FYI")["successful"] is True
    assert gate.get_action(held["action_id"]).status == "sent"


# -- expiry ---------------------------------------------------------------------------------------------------


def test_pending_actions_expire_after_seven_days(gmail, log, run_as, monkeypatch):
    run_as("alice")
    action_id = create_draft()["action_id"]
    ia_tools.send_draft(action_id=action_id)
    created = gate.get_action(action_id).created_at

    monkeypatch.setattr(gate, "_now", lambda: created + timedelta(days=7))
    assert gate.get_action(action_id).status == "pending"  # boundary: not yet

    monkeypatch.setattr(gate, "_now", lambda: created + timedelta(days=7, seconds=1))
    result = in_turn("user", created + timedelta(days=7, seconds=1), ia_tools.approve_send, action_id)
    assert not result.success and "expired" in result.payload["error"]
    assert gate.get_action(action_id).status == "expired"
    assert not ia_tools.send_draft(action_id=action_id).success


def test_approved_but_unsent_actions_expire_too(gmail, log, run_as, monkeypatch):
    run_as("alice")
    held = reply()
    ia_tools.send_draft(action_id=held["action_id"])
    user_says_after_showing(held["action_id"])
    created = gate.get_action(held["action_id"]).created_at
    monkeypatch.setattr(gate, "_now", lambda: created + timedelta(days=8))
    assert reply()["status"] == "approval_required" and gmail.sent == []
    assert gate.get_action(held["action_id"]).status == "expired"


def test_expire_stale_sweeps_old_rows(gmail, log, run_as):
    run_as("alice")
    old = create_draft()["action_id"]
    created = gate.get_action(old).created_at
    assert gate.expire_stale(created + timedelta(days=6)) == 0
    assert gate.expire_stale(created + timedelta(days=8)) == 1
    assert gate.get_approval_store().get(old).status == "expired"


# -- lifecycle open work --------------------------------------------------------------------------------------


def test_agents_with_pending_actions_are_open_work(gmail, log, run_as):
    run_as("alice")
    action_id = create_draft()["action_id"]
    created = gate.get_action(action_id).created_at
    assert gate.agent_ids_with_pending(created + timedelta(days=6)) == {"alice"}
    assert gate.agent_ids_with_pending(created + timedelta(days=8)) == set()
    ia_tools.send_draft(action_id=action_id)
    user_says_after_showing(action_id)
    assert gate.agent_ids_with_pending(created) == set()  # approved is no longer waiting on the user


class NoTriggers:
    def agent_ids_with_triggers_due(self, now, within=timedelta(hours=24)):
        return set()


def test_lifecycle_sweep_keeps_agents_with_pending_sends_active(gmail, log, run_as, monkeypatch):
    run_as("alice")
    action_id = create_draft()["action_id"]
    now = gate.get_action(action_id).created_at + timedelta(days=1)

    def card(agent_id):
        return AgentCard(id=agent_id, name=agent_id, purpose=agent_id, created_at=now - timedelta(days=60),
                         last_active_at=now - timedelta(days=40))

    directory = InMemoryDirectory([card("alice"), card("bob")])
    monkeypatch.setattr(scheduler_module, "get_agent_directory", lambda: directory)
    scheduler = TriggerScheduler(poll_interval_seconds=0)
    scheduler._service = NoTriggers()
    assert scheduler.run_lifecycle_sweep(now) == {"bob": "archived"}

    # Eight days on the pending send has expired, so it no longer holds the agent active.
    assert scheduler.run_lifecycle_sweep(now + timedelta(days=7)) == {"alice": "archived"}
    assert gate.get_approval_store().get(action_id).status == "expired"


# -- seams: the ablation switch, the decision sink, partial-result wording, schemas ---------------------------


def test_gate_off_allows_but_still_reports_the_decision(gmail, log, run_as, monkeypatch):
    decisions = []
    monkeypatch.setattr(gate, "enforce", False)
    monkeypatch.setattr(gate, "decision_sink", decisions.append)
    run_as("alice")
    result = reply()
    assert result["successful"] is True and len(gmail.sent) == 1
    assert [(d.allowed, d.would_allow, d.reason, d.enforced) for d in decisions] == [
        (True, False, "not_approved", False)
    ]


def test_gate_on_reports_blocked_decisions(gmail, log, run_as, monkeypatch):
    decisions = []
    monkeypatch.setattr(gate, "decision_sink", decisions.append)
    run_as("alice")
    reply()
    assert [(d.allowed, d.reason, d.kind, d.agent_id, d.recipients) for d in decisions] == [
        (False, "not_approved", "reply", "alice", ["dana@example.com"])
    ]


def test_held_sends_are_not_reported_as_sent_side_effects(tmp_path):
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    start = len(list(logs.iter_entries("a")))
    logs.record_action("a", 'Calling gmail_reply_to_thread with: {"recipient_email": "di@y.com", "thread_id": "t"}')
    logs.record_tool_response("a", "gmail_reply_to_thread", '{"status": "approval_required", "action_id": 4}')
    assert side_effects_since(logs, "a", start) == ["reply_to_thread held for approval (action 4)"]


def test_ia_schemas_and_dispatch_include_approve_send():
    names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    assert "approve_send" in names
    send_draft = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "send_draft")
    assert "action_id" in send_draft["function"]["parameters"]["properties"]
    result = ia_tools.handle_tool_call("approve_send", {"action_id": 12345})
    assert not result.success and "Only the user" in result.payload["error"]
