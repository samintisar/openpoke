import json

import pytest
from fastapi import HTTPException

from server.models import GmailStatusPayload
from server.routes import fake_gmail as routes
from server.services.approvals import gate
from server.services.gmail import client, fake


@pytest.fixture
def fake_mode(monkeypatch):
    monkeypatch.setenv("OPENPOKE_FAKE_GMAIL", "1")
    monkeypatch.setattr(fake, "_mailbox", None)
    monkeypatch.setattr(gate, "decision_sink", None)
    yield
    fake._mailbox = None


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("OPENPOKE_FAKE_GMAIL", raising=False)
    assert not fake.enabled()
    with pytest.raises(HTTPException) as err:
        routes._require_fake_mode()
    assert err.value.status_code == 404


def test_gmail_calls_go_to_the_seeded_mailbox(fake_mode):
    assert client.get_active_gmail_user_id() == fake.FAKE_USER_ID
    result = client.execute_gmail_tool("GMAIL_FETCH_EMAILS", "anyone", arguments={"query": "bathroom"})
    subjects = [m["subject"] for m in result["data"]["messages"]]
    assert result["successful"] and subjects == ["Bathroom remodel quote"]
    assert gate.decision_sink == fake.mailbox().record_decision


def test_status_reports_the_fake_owner(fake_mode):
    body = json.loads(client.fetch_status(GmailStatusPayload()).body)
    assert body["connected"] and body["email"] == fake.FAKE_OWNER


async def test_delivered_mail_is_new_and_shows_in_the_snapshot(fake_mode):
    before = len(fake.snapshot()["inbox"])
    sent = await routes.fake_gmail_deliver(routes.IncomingEmail(sender="Zoe <zoe@x.test>", subject="Hi", body="Lunch?"))
    message = json.loads(sent.body)["message"]
    snap = json.loads((await routes.fake_gmail_snapshot()).body)
    assert len(snap["inbox"]) == before + 1 and snap["inbox"][0]["id"] == message["id"]
    assert message["timestamp"] > max(m["timestamp"] for m in snap["inbox"][1:])
    assert snap["sent"] == [] and snap["blocked"] == []
