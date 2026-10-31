"""Biometric enrolment: authorisation, consent, template build, duplicate check, approvals.

Identity binding: consent is attested by the *subject* entering their own
account password (verified server-side, rate-limited like logins). The
operator alone can never assert consent for someone else.

Approval rules — a candidate template becomes active immediately only when
none of these apply; otherwise an EnrolmentRequest stays PENDING for an admin:
  * EXISTING_TEMPLATE   — a usable template already exists (replacement)
  * DUPLICATE_SUSPECTED — the new template matches another user's template
  * SELF_ENROLMENT      — a non-admin enrolling themselves

Tenancy: enrolments, consent records and requests belong to the actor's active
organisation. Duplicate detection only compares templates of the same
organisation — templates are never compared across tenants.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from backend.core.events import bus
from backend.core.deps import Actor, membership_role
from backend.core.permissions import Permission
from backend.models.biometric import (
    BiometricEnrolment, ConsentRecord, EnrolmentRequest, EnrolmentStatus, RequestStatus,
)
from backend.models.user import RoleEnum, User
from backend.services import alert_service, notification_service
from backend.services.audit_service import audit, log_security_event
from backend.services.auth_service import check_rate_limit, record_login_attempt, verify_password
from backend.services.biometric.templates import (
    TemplateError, build_template, cosine_distance, decrypt_vectors, encrypt_vectors,
)
from backend.services.policy_service import Policy

log = logging.getLogger(__name__)

CONSENT_VERSION = "2026-10"
CONSENT_TEXT = (
    "I consent to the capture of my facial images to create an encrypted biometric template, "
    "used only to verify my identity in this system. Images are not stored. I may withdraw consent "
    "at any time, which deletes my template."
)


class EnrolmentDenied(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def get_enrolment(db: Session, org_id: str, user_id: str) -> Optional[BiometricEnrolment]:
    return (db.query(BiometricEnrolment)
            .filter(BiometricEnrolment.organization_id == org_id, BiometricEnrolment.user_id == user_id).first())


def get_or_create_enrolment(db: Session, org_id: str, user_id: str) -> BiometricEnrolment:
    enr = get_enrolment(db, org_id, user_id)
    if enr is None:
        enr = BiometricEnrolment(organization_id=org_id, user_id=user_id, status=EnrolmentStatus.NOT_ENROLLED,
                                 template_version=2)
        db.add(enr)
        db.flush()
    return enr


def _deny(db: Session, actor: Actor, target: Optional[User], code: str, message: str, ip: str,
          status: int = 403, alert: bool = True) -> EnrolmentDenied:
    log_security_event(db, "UNAUTHORIZED_ENROLMENT_ATTEMPT", f"{actor.email}: {message}", "HIGH",
                       user_id=actor.id, ip_address=ip,
                       metadata={"target_user_id": target.id if target else None, "code": code}, commit=False,
                       org_id=actor.org_id)
    if alert:
        alert_service.raise_alert(db, "UNAUTHORIZED_ENROLMENT_ATTEMPT", "HIGH", "user", actor.id,
                                  f"{actor.email} attempted a biometric change they are not allowed to make ({code})",
                                  metadata={"target_user_id": target.id if target else None}, commit=False,
                                  org_id=actor.org_id)
    db.commit()
    return EnrolmentDenied(status, code, message)


def authorize_enrolment_actor(db: Session, actor: Actor, target: User, ip: str) -> None:
    """Server-side rules for who may create or replace whose biometric template."""
    if not actor.can(Permission.ENROL_CAPTURE):
        raise _deny(db, actor, target, "ROLE_NOT_PERMITTED", "Your role cannot modify biometric records", ip)
    target_role = membership_role(db, actor.org_id, target.id)
    if target_role is None:
        raise EnrolmentDenied(404, "TARGET_NOT_FOUND", "User not found in this organisation")
    if not target.is_active:
        raise EnrolmentDenied(409, "TARGET_DISABLED", "This user account is disabled")
    if target_role == RoleEnum.ADMIN and actor.role != RoleEnum.ADMIN:
        raise _deny(db, actor, target, "PRIVILEGED_TARGET", "Only administrators may enrol administrator accounts", ip)


def record_consent(db: Session, actor: Actor, target: User, subject_password: str, ip: str) -> BiometricEnrolment:
    self_service = actor.id == target.id
    if not self_service:
        authorize_enrolment_actor(db, actor, target, ip)
    elif membership_role(db, actor.org_id, target.id) is None:
        raise EnrolmentDenied(404, "TARGET_NOT_FOUND", "User not found in this organisation")
    allowed, _, retry = check_rate_limit(db, target.email, ip)
    if not allowed:
        raise EnrolmentDenied(429, "ATTESTATION_RATE_LIMITED", f"Too many attempts. Try again in {retry}s.")
    ok = verify_password(subject_password, target.hashed_password)
    record_login_attempt(db, target.email, ip, ok)
    if not ok:
        log_security_event(db, "CONSENT_ATTESTATION_FAILED", f"Wrong subject password for {target.email}",
                           "WARNING", user_id=actor.id, ip_address=ip, metadata={"target_user_id": target.id},
                           commit=False, org_id=actor.org_id)
        since = datetime.utcnow() - timedelta(hours=1)
        from backend.models.verification import SecurityEvent
        fails = (db.query(SecurityEvent)
                 .filter(SecurityEvent.event_type == "CONSENT_ATTESTATION_FAILED", SecurityEvent.user_id == actor.id,
                         SecurityEvent.organization_id == actor.org_id,
                         SecurityEvent.timestamp >= since).count())
        if fails >= 3:
            alert_service.raise_alert(db, "CONSENT_ATTESTATION_FAILURES", "HIGH", "user", actor.id,
                                      f"{fails + 1} failed consent attestations in the last hour", commit=False,
                                      org_id=actor.org_id)
        db.commit()
        raise EnrolmentDenied(403, "ATTESTATION_FAILED", "The subject's password was not accepted")

    enr = get_or_create_enrolment(db, actor.org_id, target.id)
    now = datetime.utcnow()
    enr.consent_given = True
    enr.consent_timestamp = now
    if enr.status == EnrolmentStatus.WITHDRAWN:
        enr.status = EnrolmentStatus.NOT_ENROLLED
    db.add(ConsentRecord(organization_id=actor.org_id, user_id=target.id, consent_given=True, consent_text=CONSENT_TEXT,
                         consent_version=CONSENT_VERSION, purpose="BIOMETRIC_ENROLMENT",
                         method="SELF" if self_service else "SUBJECT_PASSWORD",
                         actor_user_id=actor.id, ip_address=ip, recorded_at=now))
    audit(db, actor.id, "CONSENT_GRANTED", "user", target.id, {"version": CONSENT_VERSION}, ip, commit=False,
          org_id=actor.org_id)
    log_security_event(db, "CONSENT_RECORDED", f"Biometric consent granted for {target.email}", "INFO",
                       user_id=actor.id, ip_address=ip, metadata={"target_user_id": target.id}, commit=False,
                       org_id=actor.org_id)
    db.commit()
    return enr


def withdraw_consent(db: Session, actor: Actor, target: User, ip: str, reason: str = "") -> BiometricEnrolment:
    if actor.id != target.id and not actor.can(Permission.ENROL_APPROVE):
        raise _deny(db, actor, target, "WITHDRAW_NOT_PERMITTED",
                    "Only the subject or an administrator may withdraw consent", ip)
    if membership_role(db, actor.org_id, target.id) is None:
        raise EnrolmentDenied(404, "TARGET_NOT_FOUND", "User not found in this organisation")
    enr = get_or_create_enrolment(db, actor.org_id, target.id)
    now = datetime.utcnow()
    had_template = enr.embedding_encrypted is not None
    enr.consent_given = False
    enr.consent_timestamp = None
    enr.embedding_encrypted = None
    enr.status = EnrolmentStatus.WITHDRAWN
    enr.revoked_at = now
    enr.revoked_by_user_id = actor.id
    _cancel_pending_requests(db, actor.org_id, target.id, actor.id, "Consent withdrawn")
    db.add(ConsentRecord(organization_id=actor.org_id, user_id=target.id, consent_given=False, consent_text=reason[:500] or None,
                         consent_version=CONSENT_VERSION, purpose="BIOMETRIC_ENROLMENT",
                         method="SELF" if actor.id == target.id else "ADMIN", actor_user_id=actor.id,
                         ip_address=ip, recorded_at=now))
    audit(db, actor.id, "CONSENT_WITHDRAWN", "user", target.id, {"template_deleted": had_template}, ip, commit=False,
          org_id=actor.org_id)
    log_security_event(db, "CONSENT_WITHDRAWN", f"Consent withdrawn for {target.email}; template deleted",
                       "WARNING", user_id=actor.id, ip_address=ip, metadata={"target_user_id": target.id}, commit=False,
                       org_id=actor.org_id)
    db.commit()
    bus.publish("enrolment", "enrolment_changed", {"user_id": target.id, "status": enr.status}, org_id=actor.org_id)
    return enr


def ensure_capture_allowed(db: Session, actor: Actor, target: User, ip: str) -> BiometricEnrolment:
    authorize_enrolment_actor(db, actor, target, ip)
    enr = get_or_create_enrolment(db, actor.org_id, target.id)
    if not enr.consent_given:
        db.commit()
        raise EnrolmentDenied(409, "CONSENT_REQUIRED", "The subject must give consent before enrolment")
    pending = (db.query(EnrolmentRequest)
               .filter(EnrolmentRequest.organization_id == actor.org_id, EnrolmentRequest.user_id == target.id,
                       EnrolmentRequest.status == RequestStatus.PENDING).count())
    if pending:
        db.commit()
        raise EnrolmentDenied(409, "REQUEST_PENDING", "An enrolment request for this user is awaiting approval")
    db.commit()
    return enr


@dataclass
class FinalizeResult:
    request: EnrolmentRequest
    activated: bool
    reasons: list[str]


class EnrolmentQualityError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def find_duplicates(db: Session, org_id: str, vector: np.ndarray, exclude_user_id: str, threshold: float) -> list[str]:
    """Privacy-conscious duplicate check: returns user IDs whose active template matches.

    Only called during enrolment finalisation; the IDs are stored on the request
    and shown only to administrators. There is no general search endpoint.
    """
    hits: list[str] = []
    rows = (db.query(BiometricEnrolment)
            .filter(BiometricEnrolment.organization_id == org_id,
                    BiometricEnrolment.status == EnrolmentStatus.ACTIVE,
                    BiometricEnrolment.user_id != exclude_user_id,
                    BiometricEnrolment.embedding_encrypted.isnot(None),
                    BiometricEnrolment.template_version >= 2).all())
    for row in rows:
        try:
            tmpl = decrypt_vectors(row.embedding_encrypted)
            if tmpl.dim == vector.shape[0] and cosine_distance(vector, tmpl.vector) <= threshold:
                hits.append(row.user_id)
        except TemplateError:
            continue
    return hits


def finalize_capture(db: Session, actor: Actor, target: User, session_id: str, samples: list[np.ndarray],
                     quality_scores: list[float], policy: Policy, model_name: str, ip: str) -> FinalizeResult:
    built = build_template(samples)
    if built.consistency > policy.enrol_consistency_max_distance:
        raise EnrolmentQualityError(
            "INCONSISTENT_SAMPLES",
            f"Samples were not consistent enough (max distance {built.consistency:.3f} > "
            f"{policy.enrol_consistency_max_distance}). Re-capture in steady lighting.")
    enr = get_or_create_enrolment(db, actor.org_id, target.id)
    reasons: list[str] = []
    if enr.is_usable and not enr.reenrolment_requested:
        reasons.append("EXISTING_TEMPLATE")
    if actor.id == target.id and actor.role != RoleEnum.ADMIN:
        reasons.append("SELF_ENROLMENT")
    duplicates: list[str] = []
    if policy.duplicate_check_enabled:
        duplicates = find_duplicates(db, actor.org_id, built.vector, target.id, policy.match_threshold)
        if duplicates:
            reasons.append("DUPLICATE_SUSPECTED")

    quality = float(np.mean(quality_scores)) if quality_scores else 0.0
    req = EnrolmentRequest(
        organization_id=actor.org_id, user_id=target.id, requested_by_user_id=actor.id, session_id=session_id,
        kind="REENROL" if enr.embedding_encrypted is not None else "INITIAL",
        status=RequestStatus.PENDING if reasons else RequestStatus.AUTO_APPROVED,
        approval_reasons=json.dumps(reasons), candidate_template_encrypted=encrypt_vectors([built.vector], model_name),
        embedding_dim=int(built.vector.shape[0]), model_name=model_name, sample_count=built.sample_count,
        quality_score=round(quality, 4), consistency_score=round(built.consistency, 4),
        duplicate_user_ids=json.dumps(duplicates) if duplicates else None,
    )
    db.add(req)
    db.flush()
    audit(db, actor.id, "ENROLMENT_CAPTURED", "user", target.id,
          {"request_id": req.id, "status": req.status, "reasons": reasons, "samples": built.sample_count}, ip,
          commit=False, org_id=actor.org_id)
    if not reasons:
        _activate(db, enr, req, approver_id=None)
        log_security_event(db, "ENROLMENT_COMPLETE", f"{target.email} enrolled ({built.sample_count} samples)",
                           "INFO", user_id=actor.id, ip_address=ip, metadata={"target_user_id": target.id},
                           commit=False, org_id=actor.org_id)
    else:
        log_security_event(db, "ENROLMENT_PENDING_APPROVAL",
                           f"Enrolment for {target.email} requires approval: {', '.join(reasons)}", "WARNING",
                           user_id=actor.id, ip_address=ip, metadata={"target_user_id": target.id}, commit=False,
                           org_id=actor.org_id)
        if "EXISTING_TEMPLATE" in reasons:
            alert_service.raise_alert(db, "TEMPLATE_REPLACEMENT_REQUESTED", "MEDIUM", "user", target.id,
                                      f"{actor.email} captured a replacement template for {target.email}",
                                      window_minutes=24 * 60, commit=False, org_id=actor.org_id)
            since = datetime.utcnow() - timedelta(hours=24)
            n = (db.query(EnrolmentRequest)
                 .filter(EnrolmentRequest.organization_id == actor.org_id,
                         EnrolmentRequest.user_id == target.id, EnrolmentRequest.kind == "REENROL",
                         EnrolmentRequest.created_at >= since).count())
            if n >= 2:
                alert_service.raise_alert(db, "UNUSUAL_TEMPLATE_REPLACEMENT", "HIGH", "user", target.id,
                                          f"{n} template replacement attempts for {target.email} in 24h",
                                          window_minutes=24 * 60, commit=False, org_id=actor.org_id)
        if duplicates:
            alert_service.raise_alert(db, "DUPLICATE_ENROLMENT_SUSPECTED", "HIGH", "user", target.id,
                                      f"New template for {target.email} matches {len(duplicates)} other enrolled user(s)",
                                      metadata={"request_id": req.id}, window_minutes=24 * 60, commit=False,
                                      org_id=actor.org_id)
    db.commit()
    bus.publish("enrolment", "enrolment_changed", {"user_id": target.id, "request_status": req.status},
                org_id=actor.org_id)
    return FinalizeResult(request=req, activated=not reasons, reasons=reasons)


def _activate(db: Session, enr: BiometricEnrolment, req: EnrolmentRequest, approver_id: Optional[str]) -> None:
    now = datetime.utcnow()
    enr.embedding_encrypted = req.candidate_template_encrypted
    enr.template_version = 2
    enr.embedding_dim = req.embedding_dim
    enr.model_name = req.model_name
    enr.sample_count = req.sample_count
    enr.quality_score = req.quality_score
    enr.consistency_score = req.consistency_score
    enr.status = EnrolmentStatus.ACTIVE
    enr.enrolled_at = now
    enr.enrolled_by_user_id = req.requested_by_user_id
    enr.approved_by_user_id = approver_id
    enr.reenrolment_requested = False
    enr.revoked_at = None
    enr.revoked_by_user_id = None
    req.candidate_template_encrypted = None   # single copy of the template: on the enrolment row


def review_request(db: Session, actor: Actor, request_id: str, approve: bool, note: str, policy: Policy,
                   ip: str) -> EnrolmentRequest:
    if not actor.can(Permission.ENROL_APPROVE):
        raise _deny(db, actor, None, "APPROVE_NOT_PERMITTED", "Only administrators may review enrolment requests", ip)
    req = (db.query(EnrolmentRequest)
           .filter(EnrolmentRequest.id == request_id, EnrolmentRequest.organization_id == actor.org_id).first())
    if req is None:
        raise EnrolmentDenied(404, "NOT_FOUND", "Request not found")
    if req.status != RequestStatus.PENDING:
        raise EnrolmentDenied(409, "NOT_PENDING", f"Request is already {req.status}")
    if policy.require_four_eyes and req.requested_by_user_id == actor.id:
        raise EnrolmentDenied(409, "FOUR_EYES", "Policy requires a different administrator to review this request")
    enr = get_or_create_enrolment(db, actor.org_id, req.user_id)
    now = datetime.utcnow()
    req.reviewed_by_user_id, req.reviewed_at, req.review_note = actor.id, now, (note or "")[:1000] or None
    if approve:
        if not enr.consent_given:
            raise EnrolmentDenied(409, "CONSENT_REQUIRED", "Subject consent is no longer in place")
        req.status = RequestStatus.APPROVED
        _activate(db, enr, req, approver_id=actor.id)
    else:
        req.status = RequestStatus.REJECTED
        req.candidate_template_encrypted = None
    action = "ENROLMENT_APPROVED" if approve else "ENROLMENT_REJECTED"
    audit(db, actor.id, action, "enrolment_request", req.id, {"user_id": req.user_id, "note": note[:200]}, ip,
          commit=False, org_id=actor.org_id)
    log_security_event(db, action, f"Request {req.id[:8]} for user {req.user_id[:8]} {'approved' if approve else 'rejected'}",
                       "WARNING", user_id=actor.id, ip_address=ip, metadata={"target_user_id": req.user_id},
                       commit=False, org_id=actor.org_id)
    for recipient in {req.requested_by_user_id, req.user_id} - {None, actor.id}:
        notification_service.notify(db, actor.org_id, recipient, action,
                                    "Enrolment approved" if approve else "Enrolment rejected",
                                    "A biometric enrolment request was " + ("approved." if approve else "rejected."),
                                    link=f"/users/{req.user_id}")
    db.commit()
    bus.publish("enrolment", "enrolment_changed", {"user_id": req.user_id, "request_status": req.status},
                org_id=actor.org_id)
    return req


def _cancel_pending_requests(db: Session, org_id: str, user_id: str, actor_id: str, note: str) -> None:
    for req in (db.query(EnrolmentRequest)
                .filter(EnrolmentRequest.organization_id == org_id, EnrolmentRequest.user_id == user_id,
                        EnrolmentRequest.status == RequestStatus.PENDING).all()):
        req.status = RequestStatus.CANCELLED
        req.candidate_template_encrypted = None
        req.reviewed_by_user_id, req.reviewed_at, req.review_note = actor_id, datetime.utcnow(), note


def revoke(db: Session, actor: Actor, target: User, reason: str, ip: str) -> BiometricEnrolment:
    if not actor.can(Permission.ENROL_REVOKE):
        raise _deny(db, actor, target, "REVOKE_NOT_PERMITTED", "Only administrators may revoke enrolments", ip)
    enr = get_enrolment(db, actor.org_id, target.id)
    if enr is None or enr.embedding_encrypted is None:
        raise EnrolmentDenied(404, "NOT_ENROLLED", "No template to revoke")
    enr.embedding_encrypted = None
    enr.status = EnrolmentStatus.REVOKED
    enr.revoked_at = datetime.utcnow()
    enr.revoked_by_user_id = actor.id
    enr.reenrolment_requested = False
    _cancel_pending_requests(db, actor.org_id, target.id, actor.id, "Enrolment revoked")
    audit(db, actor.id, "ENROLMENT_REVOKED", "user", target.id, {"reason": reason[:300]}, ip, commit=False,
          org_id=actor.org_id)
    log_security_event(db, "ENROLMENT_REVOKED", f"Template for {target.email} revoked", "WARNING",
                       user_id=actor.id, ip_address=ip, metadata={"target_user_id": target.id}, commit=False,
                       org_id=actor.org_id)
    db.commit()
    bus.publish("enrolment", "enrolment_changed", {"user_id": target.id, "status": enr.status}, org_id=actor.org_id)
    return enr


def request_reenrolment(db: Session, actor: Actor, target: User, reason: str, ip: str) -> BiometricEnrolment:
    if not actor.can(Permission.ENROL_APPROVE):
        raise _deny(db, actor, target, "REQUEST_NOT_PERMITTED", "Only administrators may request re-enrolment", ip)
    enr = get_or_create_enrolment(db, actor.org_id, target.id)
    enr.reenrolment_requested = True
    if membership_role(db, actor.org_id, target.id) is None:
        raise EnrolmentDenied(404, "TARGET_NOT_FOUND", "User not found in this organisation")
    audit(db, actor.id, "REENROLMENT_REQUESTED", "user", target.id, {"reason": reason[:300]}, ip, commit=False,
          org_id=actor.org_id)
    db.commit()
    return enr


def enrolment_status_view(enr: Optional[BiometricEnrolment]) -> dict:
    if enr is None:
        return {"status": EnrolmentStatus.NOT_ENROLLED, "is_enrolled": False, "consent_given": False,
                "usable": False, "reenrolment_requested": False}
    return {
        "status": enr.status,
        "is_enrolled": enr.embedding_encrypted is not None,
        "usable": enr.is_usable,
        "consent_given": enr.consent_given,
        "consent_timestamp": enr.consent_timestamp,
        "enrolled_at": enr.enrolled_at,
        "template_version": enr.template_version,
        "model_name": enr.model_name if enr.embedding_encrypted is not None else None,
        "sample_count": enr.sample_count,
        "quality_score": enr.quality_score,
        "consistency_score": enr.consistency_score,
        "reenrolment_requested": enr.reenrolment_requested,
        "enrolled_by_user_id": enr.enrolled_by_user_id,
        "approved_by_user_id": enr.approved_by_user_id,
        "revoked_at": enr.revoked_at,
    }
