"""The end-to-end eval's in-memory Gmail; it lives in the server (`server/services/gmail/fake_mailbox.py`)."""

from server.services.gmail.fake_mailbox import (  # noqa: F401
    OWNER,
    VIRTUAL_START,
    Contact,
    Draft,
    FakeGmail,
    Message,
    SentMail,
    addresses,
)

__all__ = ["Contact", "Draft", "FakeGmail", "Message", "OWNER", "SentMail", "VIRTUAL_START", "addresses"]
