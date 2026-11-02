"""The email cleaner decodes Gmail's base64url bodies, which arrive without `=` padding."""

import base64

from server.services.gmail.processing import EmailTextCleaner


def _unpadded(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def test_unpadded_plain_body_is_decoded():
    data = _unpadded("Can we move the call to Friday?")
    assert "=" not in data and len(data) % 4
    message = {"payload": {"body": {"data": data}}, "textBody": "fallback"}
    assert EmailTextCleaner().clean_email_content(message) == "Can we move the call to Friday?"


def test_unpadded_html_part_is_decoded():
    data = _unpadded("<p>Contract approved by legal.</p>")
    message = {"payload": {"parts": [{"mimeType": "text/html", "body": {"data": data}}]}, "htmlBody": None}
    assert "Contract approved by legal." in EmailTextCleaner().clean_email_content(message)


def test_padded_body_still_decodes():
    data = base64.urlsafe_b64encode(b"Thanks!").decode("ascii")
    assert data.endswith("=")
    assert EmailTextCleaner().clean_email_content({"payload": {"body": {"data": data}}}) == "Thanks!"
