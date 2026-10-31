"""Server-side suspicious-activity detection.

Rules run inline at the point an event happens (no polling). An alert for the
same (rule, subject) that is still open and was seen within its window is
updated (occurrence count, last_seen) instead of duplicated.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.core.events import bus
from backend.models.verification import SecurityAlert, VerificationSession
from backend.services.policy_service import Policy

log = logging.getLogger(__name__)

RULES = {
    "REPEATED_VERIFICATION_FAILURES": "Repeated verification failures",
    "REPEATED_LIVENESS_FAILURES": "Repeated liveness failures",
    "ABNORMAL_VERIFICATION_FREQUENCY": "Abnormal verification frequency",
    "EXCESSIVE_SESSION_CREATION": "Excessive session creation",
    "SESSION_RATE_LIMITED": "Operator hit the session rate limit",
    "UNAUTHORIZED_ENROLMENT_ATTEMPT": "Unauthorised enrolment attempt",
    "TEMPLATE_REPLACEMENT_REQUESTED": "Biometric template replacement requested",
    "UNUSUAL_TEMPLATE_REPLACEMENT": "Unusual template replacement activity",
    "DUPLICATE_ENROLMENT_SUSPECTED": "Possible duplicate enrolment",
    "SESSION_ACCESS_VIOLATION": "Session access violation",
    "FRAME_REPLAY_DETECTED": "Replayed camera frame",
    "ACCOUNT_LOCKOUT": "Account locked after failed logins",
    "REFRESH_TOKEN_REUSE": "Refresh-token reuse (possible session theft)",
    "CONSENT_ATTESTATION_FAILURES": "Repeated failed consent attestations",
    "CROSS_TENANT_ACCESS": "Cross-organisation access attempt",
    "PAIRING_ABUSE": "Suspicious QR pairing activity",
    "PASSKEY_FAILURES": "Repeated passkey failures",
    "MFA_FAILURES": "Repeated second-factor failures",
    "REVOKED_DEVICE_ACCESS": "Revoked device attempted access",
    "DEVICE_SIGNATURE_INVALID": "Invalid device signature",
    "OFFLINE_AUDIT_INTEGRITY": "Offline audit chain integrity failure",
    "API_CREDENTIAL_ABUSE": "Integration credential abuse",
}


def raise_alert(db: Session, rule: str, severity: str, subject_type: Optional[str], subject_id: Optional[str],
                description: str, window_minutes: int = 60, metadata: Optional[dict[str, Any]] = None,
                commit: bool = True, org_id: Optional[str] = None) -> SecurityAlert:
    now = datetime.utcnow()
    existing = (
        db.query(SecurityAlert)
        .filter(SecurityAlert.rule == rule, SecurityAlert.subject_id == subject_id,
                SecurityAlert.organization_id == org_id,
                SecurityAlert.status != "RESOLVED",
                SecurityAlert.last_seen_at >= now - timedelta(minutes=window_minutes))
        .order_by(SecurityAlert.last_seen_at.desc())
        .first()
    )
    if existing:
        existing.occurrence_count += 1
        existing.last_seen_at = now
        existing.description = description[:1000]
        if _rank(severity) > _rank(existing.severity):
            existing.severity = severity
        alert = existing
        created = False
    else:
        alert = SecurityAlert(organization_id=org_id, rule=rule, severity=severity, status="OPEN", subject_type=subject_type,
                              subject_id=subject_id, title=RULES.get(rule, rule), description=description[:1000],
                              first_seen_at=now, last_seen_at=now,
                              metadata_json=json.dumps(metadata, default=str) if metadata else None)
        db.add(alert)
        created = True
    if commit:
        db.commit()
    log.warning("[ALERT] %s %s subject=%s:%s", severity, rule, subject_type, subject_id)
    bus.publish("alerts", "alert", {"id": alert.id, "rule": rule, "severity": alert.severity,
                                    "title": alert.title, "created": created,
                                    "count": alert.occurrence_count}, org_id=org_id)
    if created and org_id:
        from backend.services import notification_service
        notification_service.notify_permission(db, org_id, "SUSPICIOUS_ACTIVITY", f"Security alert: {alert.title}",
                                               description[:300], link="/alerts", severity="WARNING",
                                               permission="alerts:manage")
    return alert


def raise_for_user(db: Session, user_id: Optional[str], rule: str, severity: str, description: str,
                   window_minutes: int = 60) -> int:
    """Account-level alert (no organisation context): raised in every organisation the user belongs to."""
    if not user_id:
        return 0
    from backend.models.organization import Membership
    orgs = [m.organization_id for m in db.query(Membership).filter(Membership.user_id == user_id).all()]
    for org_id in orgs:
        raise_alert(db, rule, severity, "user", user_id, description, window_minutes=window_minutes, org_id=org_id)
    return len(orgs)


def _rank(sev: str) -> int:
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(sev, 0)


def _count_outcomes(db: Session, org_id: str, target_user_id: str, outcome: str, since: datetime) -> int:
    return (db.query(VerificationSession)
            .filter(VerificationSession.organization_id == org_id,
                    VerificationSession.target_user_id == target_user_id,
                    VerificationSession.purpose == "VERIFY",
                    VerificationSession.final_outcome == outcome,
                    VerificationSession.completed_at >= since)
            .count())


def evaluate_terminal_session(db: Session, session: VerificationSession, policy: Policy) -> None:
    if not session.target_user_id or session.purpose != "VERIFY":
        return
    since = datetime.utcnow() - timedelta(hours=1)
    outcome = session.final_outcome
    if outcome == "NOT_MATCHED":
        n = _count_outcomes(db, session.organization_id, session.target_user_id, "NOT_MATCHED", since)
        if n >= policy.alert_failed_verifications_threshold:
            raise_alert(db, "REPEATED_VERIFICATION_FAILURES", "HIGH", "user", session.target_user_id,
                        f"{n} failed identity matches for this user in the last hour", commit=False,
                        org_id=session.organization_id)
    elif outcome == "LIVENESS_FAILED":
        n = _count_outcomes(db, session.organization_id, session.target_user_id, "LIVENESS_FAILED", since)
        if n >= policy.alert_liveness_failures_threshold:
            raise_alert(db, "REPEATED_LIVENESS_FAILURES", "HIGH", "user", session.target_user_id,
                        f"{n} liveness failures for this user in the last hour", commit=False,
                        org_id=session.organization_id)
    elif outcome == "VERIFIED":
        n = _count_outcomes(db, session.organization_id, session.target_user_id, "VERIFIED", since)
        if n >= policy.alert_verification_frequency_threshold:
            raise_alert(db, "ABNORMAL_VERIFICATION_FREQUENCY", "MEDIUM", "user", session.target_user_id,
                        f"{n} successful verifications for this user in the last hour", commit=False,
                        org_id=session.organization_id)


def evaluate_session_creation(db: Session, org_id: str, operator_id: str, recent_count: int, policy: Policy) -> None:
    if recent_count >= policy.alert_session_creation_threshold:
        raise_alert(db, "EXCESSIVE_SESSION_CREATION", "MEDIUM", "user", operator_id,
                    f"{recent_count} capture sessions created in 10 minutes", window_minutes=10, commit=False,
                    org_id=org_id)
