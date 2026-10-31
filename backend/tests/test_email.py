"""Email delivery against a real local SMTP server (aiosmtpd): notifications, OTP codes, invitations."""
from __future__ import annotations

import re
import socket
from datetime import datetime, timedelta
from email import message_from_bytes

import pyotp
import pytest
from aiosmtpd.controller import Controller

from backend.config import settings
from backend.database import SessionLocal
from backend.models.mfa import AuthChallenge
from backend.models.reviews import Notification
from backend.services import notification_service
from backend.tests.helpers import PASSWORD, admin_headers, default_org_id, login, make_user
from backend.tests.test_passkeys_mfa import API, step_up_password, totp_enable


class _Inbox:
    def __init__(self):
        self.messages = []

    async def handle_DATA(self, server, session, envelope):
        self.messages.append(message_from_bytes(envelope.original_content))
        return "250 OK"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def smtp(monkeypatch):
    inbox = _Inbox()
    port = _free_port()
    ctl = Controller(inbox, hostname="127.0.0.1", port=port)
    ctl.start()
    monkeypatch.setattr(settings, "smtp_host", "127.0.0.1")
    monkeypatch.setattr(settings, "smtp_port", port)
    monkeypatch.setattr(settings, "smtp_starttls", False)
    yield inbox
    ctl.stop()


def _body(msg) -> str:
    return msg.get_payload(decode=True).decode() if not msg.is_multipart() else msg.get_payload(0).get_payload(decode=True).decode()


def test_notification_email_is_delivered_only_when_opted_in(client, smtp):
    user = make_user(name="Mail User")
    db = SessionLocal()
    notification_service.set_preference(db, user.id, "VERIFICATION_COMPLETED", True, True)
    db.commit()
    a = notification_service.notify(db, default_org_id(), user.id, "VERIFICATION_COMPLETED", "Verified", "Done")
    b = notification_service.notify(db, default_org_id(), user.id, "LIVENESS_FAILED", "Liveness", "No email wanted")
    db.commit()
    assert (a.email_status, b.email_status) == ("PENDING", "NOT_REQUESTED")
    stats = notification_service.deliver_pending_emails(db)
    db.close()
    assert stats["sent"] == 1 and len(smtp.messages) == 1
    msg = smtp.messages[0]
    assert msg["To"] == user.email and "Verified" in msg["Subject"]
    db = SessionLocal()
    assert db.query(Notification).filter(Notification.id == a.id).one().email_status == "SENT"
    db.close()


def test_unreachable_smtp_retries_with_backoff_then_fails(client, monkeypatch):
    monkeypatch.setattr(settings, "smtp_host", "127.0.0.1")
    monkeypatch.setattr(settings, "smtp_port", _free_port())      # nothing listening
    monkeypatch.setattr(settings, "smtp_starttls", False)
    monkeypatch.setattr(settings, "smtp_timeout_seconds", 1.0)
    user = make_user()
    db = SessionLocal()
    n = notification_service.notify(db, default_org_id(), user.id, "ACCOUNT_SECURITY", "Security", "x")
    db.commit()
    nid = n.id
    for attempt in range(1, 6):
        db.query(Notification).filter(Notification.id == nid).update({"email_next_attempt_at": datetime.utcnow()})
        db.commit()
        notification_service.deliver_pending_emails(db)
        row = db.query(Notification).filter(Notification.id == nid).one()
        db.refresh(row)
        assert row.email_attempts == attempt
        if attempt < 5:
            assert row.email_status == "PENDING" and row.email_next_attempt_at > datetime.utcnow()
    assert row.email_status == "FAILED" and row.email_last_error
    db.close()


def test_without_smtp_email_is_recorded_unavailable(client):
    user = make_user()
    db = SessionLocal()
    n = notification_service.notify(db, default_org_id(), user.id, "ACCOUNT_SECURITY", "Security", "x")
    db.commit()
    assert n.email_status == "UNAVAILABLE"
    db.close()


def test_email_otp_sign_in_and_expiry(client, smtp):
    user = make_user()
    totp_enable(client, step_up_password(client, login(client, user.email)))
    r = client.post(f"{API}/login", json={"email": user.email, "password": PASSWORD})
    assert "email_otp" in r.json()["methods"]
    token = r.json()["pending_token"]
    assert client.post(f"{API}/mfa/email-otp", json={"pending_token": token}).status_code == 200
    code = re.search(r"\b(\d{6})\b", _body(smtp.messages[-1])).group(1)
    r = client.post(f"{API}/mfa/verify", json={"pending_token": token, "method": "email_otp", "code": code})
    assert r.status_code == 200 and r.json()["user"]["amr"] == ["pwd", "otp"]
    client.cookies.clear()

    # An expired code is refused.
    token = client.post(f"{API}/login", json={"email": user.email, "password": PASSWORD}).json()["pending_token"]
    client.post(f"{API}/mfa/email-otp", json={"pending_token": token})
    code = re.search(r"\b(\d{6})\b", _body(smtp.messages[-1])).group(1)
    db = SessionLocal()
    db.query(AuthChallenge).filter(AuthChallenge.kind == "EMAIL_OTP", AuthChallenge.consumed_at.is_(None)).update(
        {"expires_at": datetime.utcnow() - timedelta(seconds=1)})
    db.commit()
    db.close()
    r = client.post(f"{API}/mfa/verify", json={"pending_token": token, "method": "email_otp", "code": code})
    assert r.status_code == 401


def test_invitation_email_is_sent(client, smtp):
    r = client.post("/api/v1/organizations/current/invitations", json={"email": "invitee@test.local", "role": "VIEWER"},
                    headers=admin_headers(client))
    assert r.status_code == 201 and r.json()["email_status"] == "SENT"
    assert smtp.messages[-1]["To"] == "invitee@test.local"
    assert r.json()["link"].rsplit("/", 1)[1] in _body(smtp.messages[-1])
