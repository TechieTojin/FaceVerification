"""Outbound email over SMTP.

If SMTP_HOST is not configured, email is *unavailable*: callers record that
state explicitly and nothing pretends a message was sent.
"""
from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage

from backend.config import settings

log = logging.getLogger(__name__)


class EmailUnavailable(RuntimeError):
    pass


class EmailSendError(RuntimeError):
    pass


def available() -> bool:
    return settings.email_available


def send(to: str, subject: str, body: str) -> None:
    if not available():
        raise EmailUnavailable("No SMTP server is configured")
    msg = EmailMessage()
    msg["From"] = settings.smtp_from
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=settings.smtp_timeout_seconds) as smtp:
            if settings.smtp_starttls:
                smtp.starttls()
            if settings.smtp_username:
                smtp.login(settings.smtp_username, settings.smtp_password)
            refused = smtp.send_message(msg)
    except (OSError, smtplib.SMTPException) as exc:
        raise EmailSendError(f"{type(exc).__name__}: {exc}"[:300]) from exc
    if refused:
        raise EmailSendError(f"Recipient refused: {', '.join(refused)}"[:300])
    log.info("Email sent (subject=%r)", subject)
