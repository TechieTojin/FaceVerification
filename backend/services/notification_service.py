"""Notification centre: per-user, per-organisation notifications with optional email delivery.

* Rows are created in the caller's transaction (`commit` is the caller's job)
  and pushed live on the user-addressed `notifications` SSE channel.
* Email is queued (`PENDING`) only when the recipient opted in AND SMTP is
  configured; otherwise the row records `UNAVAILABLE`. Delivery happens in the
  maintenance loop with exponential back-off and is never reported as sent
  unless the SMTP server accepted the message.
* Bodies never contain biometric data (no scores, templates or images).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from backend.core.events import bus
from backend.core.permissions import Permission, permissions_for
from backend.models.organization import Membership, MembershipStatus
from backend.models.reviews import Notification, NotificationPreference
from backend.models.user import User
from backend.services import email_service

log = logging.getLogger(__name__)

# type: (label, default in-app, default email, mandatory in-app)
TYPES: dict[str, tuple[str, bool, bool, bool]] = {
    "VERIFICATION_COMPLETED": ("Verification completed", True, False, False),
    "VERIFICATION_REJECTED": ("Verification rejected", True, False, False),
    "LIVENESS_FAILED": ("Liveness check failed", True, False, False),
    "SUSPICIOUS_ACTIVITY": ("Suspicious activity", True, True, False),
    "ENROLMENT_APPROVED": ("Enrolment approved", True, False, False),
    "ENROLMENT_REJECTED": ("Enrolment rejected", True, False, False),
    "DEVICE_REGISTERED": ("New device registered", True, False, False),
    "DEVICE_REVOKED": ("Device revoked", True, True, False),
    "PAIRING_EXPIRED": ("QR pairing expired", True, False, False),
    "REVIEW_ASSIGNED": ("Manual review assigned", True, True, False),
    "REVIEW_COMPLETED": ("Review completed", True, False, False),
    "ACCOUNT_SECURITY": ("Account security change", True, True, True),
}

MAX_EMAIL_ATTEMPTS = 5
_BACKOFF_MINUTES = (1, 5, 15, 60)


def preferences_for(db: Session, user_id: str) -> dict[str, dict[str, bool]]:
    rows = {p.type: p for p in db.query(NotificationPreference).filter(NotificationPreference.user_id == user_id)}
    out = {}
    for t, (label, d_app, d_email, mandatory) in TYPES.items():
        p = rows.get(t)
        out[t] = {"label": label, "in_app": True if mandatory else (p.in_app if p else d_app),
                  "email": p.email if p else d_email, "mandatory": mandatory}
    return out


def set_preference(db: Session, user_id: str, type_: str, in_app: bool, email: bool) -> None:
    if type_ not in TYPES:
        raise ValueError(f"Unknown notification type {type_}")
    if TYPES[type_][3]:
        in_app = True
    p = (db.query(NotificationPreference)
         .filter(NotificationPreference.user_id == user_id, NotificationPreference.type == type_).first())
    if p is None:
        p = NotificationPreference(user_id=user_id, type=type_)
        db.add(p)
    p.in_app, p.email = in_app, email


def notify(db: Session, org_id: Optional[str], user_id: str, type_: str, title: str, body: str = "",
           link: Optional[str] = None, severity: str = "INFO") -> Optional[Notification]:
    if type_ not in TYPES:
        raise ValueError(f"Unknown notification type {type_}")
    prefs = preferences_for(db, user_id)[type_]
    if not prefs["in_app"] and not prefs["email"]:
        return None
    email_status = "NOT_REQUESTED"
    if prefs["email"]:
        email_status = "PENDING" if email_service.available() else "UNAVAILABLE"
    n = Notification(organization_id=org_id, user_id=user_id, type=type_, severity=severity, title=title[:200],
                     body=body[:2000] or None, link=link, created_at=datetime.utcnow(), email_status=email_status,
                     email_next_attempt_at=datetime.utcnow() if email_status == "PENDING" else None,
                     read_at=None if prefs["in_app"] else datetime.utcnow())
    db.add(n)
    db.flush()
    if prefs["in_app"] and org_id:
        bus.publish("notifications", "notification", {"id": n.id, "type": type_, "title": n.title,
                                                       "severity": severity, "link": link},
                    org_id=org_id, user_id=user_id)
    return n


def notify_permission(db: Session, org_id: str, type_: str, title: str, body: str = "", link: Optional[str] = None,
                      severity: str = "INFO", permission: str = Permission.ALERTS_MANAGE.value,
                      exclude_user_id: Optional[str] = None) -> int:
    """Notify every active member of `org_id` whose role grants `permission`."""
    perm = Permission(permission)
    members = (db.query(Membership).join(User, User.id == Membership.user_id)
               .filter(Membership.organization_id == org_id, Membership.status == MembershipStatus.ACTIVE,
                       User.is_active == True).all())  # noqa: E712
    n = 0
    for m in members:
        if perm in permissions_for(m.role) and m.user_id != exclude_user_id:
            if notify(db, org_id, m.user_id, type_, title, body, link, severity):
                n += 1
    return n


def unread_count(db: Session, org_id: str, user_id: str) -> int:
    return (db.query(Notification)
            .filter(Notification.user_id == user_id, Notification.organization_id == org_id,
                    Notification.read_at.is_(None)).count())


def deliver_pending_emails(db: Session, limit: int = 20) -> dict[str, int]:
    """Send queued notification emails. Called from the maintenance loop."""
    now = datetime.utcnow()
    rows = (db.query(Notification)
            .filter(Notification.email_status == "PENDING", Notification.email_next_attempt_at <= now)
            .order_by(Notification.email_next_attempt_at).limit(limit).all())
    stats = {"sent": 0, "failed": 0, "retry": 0}
    for n in rows:
        user = db.query(User).filter(User.id == n.user_id).first()
        if user is None or not user.is_active:
            n.email_status, n.email_last_error = "FAILED", "Recipient unavailable"
            stats["failed"] += 1
            continue
        n.email_attempts += 1
        try:
            email_service.send(user.email, f"[FaceVerification] {n.title}",
                               f"{n.title}\n\n{n.body or ''}\n\nOpen FaceVerification to see details.")
            n.email_status, n.email_sent_at, n.email_last_error = "SENT", datetime.utcnow(), None
            stats["sent"] += 1
        except email_service.EmailUnavailable:
            n.email_status, n.email_last_error = "UNAVAILABLE", "No SMTP server is configured"
            stats["failed"] += 1
        except email_service.EmailSendError as exc:
            n.email_last_error = str(exc)[:300]
            if n.email_attempts >= MAX_EMAIL_ATTEMPTS:
                n.email_status = "FAILED"
                stats["failed"] += 1
            else:
                delay = _BACKOFF_MINUTES[min(n.email_attempts - 1, len(_BACKOFF_MINUTES) - 1)]
                n.email_next_attempt_at = datetime.utcnow() + timedelta(minutes=delay)
                stats["retry"] += 1
    db.commit()
    return stats


def maintenance_task(db: Session, now: datetime, policies: dict) -> dict[str, int]:
    stats = deliver_pending_emails(db)
    removed = 0
    for org_id, policy in policies.items():
        removed += (db.query(Notification)
                    .filter(Notification.organization_id == org_id, Notification.read_at.isnot(None),
                            Notification.created_at < now - timedelta(days=int(policy.retention_notifications_days)))
                    .delete(synchronize_session=False))
    return {"emails_sent": stats["sent"], "emails_failed": stats["failed"], "notifications_deleted": removed}
