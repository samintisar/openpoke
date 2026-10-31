from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.gmail import fake

router = APIRouter(prefix="/fake-gmail", tags=["fake-gmail"])


class IncomingEmail(BaseModel):
    sender: str = Field(min_length=3)
    subject: str = ""
    body: str = Field(min_length=1)
    thread_id: Optional[str] = None


def _require_fake_mode() -> None:
    if not fake.enabled():
        raise HTTPException(status_code=404, detail="Fake mailbox is off. Start the server with OPENPOKE_FAKE_GMAIL=1.")


@router.get("")
# The fake mailbox: inbox, drafts, sent mail and sends the gate blocked
async def fake_gmail_snapshot() -> JSONResponse:
    _require_fake_mode()
    return JSONResponse({"ok": True, **fake.snapshot()})


@router.post("/messages")
# Deliver an email to the fake inbox, as if someone outside had sent it
async def fake_gmail_deliver(payload: IncomingEmail) -> JSONResponse:
    _require_fake_mode()
    message = fake.deliver(sender=payload.sender, subject=payload.subject, body=payload.body,
                           thread_id=payload.thread_id)
    return JSONResponse({"ok": True, "message": message})
