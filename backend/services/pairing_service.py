"""Cross-device capture: desktop QR pairing → phone verification, and hosted integration links.

Protocol (QR):
  1. The operator creates a pairing for a subject. The QR encodes only
     `{PUBLIC_BASE_URL}/m/{token}` — an opaque 192-bit one-time token (hash stored).
  2. The phone opens the link and sees which organisation/operator asks, and a
     masked subject name. Claiming is one atomic UPDATE (WAITING → AWAITING_CONFIRMATION):
     a second claim, a reused or expired token, or a cancelled pairing fails.
  3. The claim returns a *pairing secret* (header credential for all further phone
     requests) and a 6-digit code shown on the phone.
  4. The operator types that code on the desktop. Only then can the phone start
     capture — a stolen/photographed QR claimed by someone else cannot proceed
     without the operator confirming the code the attacker's phone shows.
  5. The phone runs the normal server-side challenge; the desktop receives live
     progress on the user-addressed `pairing` SSE channel. One pairing yields one
     final decision (retries only after a non-final ending, up to policy).
Scanning a QR proves nothing about the phone; the phone is trusted only for the
capture it is explicitly paired for, and its frames get the same server checks.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.config import settings
from backend.core import crypto
from backend.core.deps import Actor, membership_role
from backend.core.errors import ApiError
from backend.core.events import bus
from backend.core.permissions import Permission
from backend.models.auth import LoginAttempt
from backend.models.organization import Organization
from backend.models.pairing import CapturePairing, OPEN_PAIRING_STATUSES, PairingKind, PairingStatus
from backend.models.user import User
from backend.models.verification import SessionPurpose, VerificationSession
from backend.services import alert_service, capture_service, notification_service
from backend.services.audit_service import audit, log_security_event
from backend.services.policy_service import get_active_policy

log = logging.getLogger(__name__)

MAX_CONFIRM_ATTEMPTS = 3
CLAIMS_PER_IP_PER_10MIN = 30
CAPTURE_WINDOW = timedelta(minutes=10)
FINAL_OUTCOMES = frozenset({"VERIFIED", "NOT_MATCHED", "LIVENESS_FAILED", "ATTEMPTS_EXHAUSTED", "ERROR"})


def _url(raw: str) -> str:
    return f"{settings.public_base_url.rstrip('/')}/m/{raw}"


def masked_name(full_name: str) -> str:
    parts = [p for p in (full_name or "").split() if p]
    if not parts:
        return "the account holder"
    return parts[0] + (f" {parts[-1][0]}." if len(parts) > 1 else "")


def _publish(p: CapturePairing, event: str, extra: Optional[dict] = None) -> None:
    if p.created_by_user_id:
        bus.publish("pairing", event, {"pairing_id": p.id, "status": p.status, **(extra or {})},
                    org_id=p.organization_id, user_id=p.created_by_user_id)


def _expire_if_needed(db: Session, p: CapturePairing) -> bool:
    if p.status in (PairingStatus.WAITING, PairingStatus.AWAITING_CONFIRMATION) and p.expires_at < datetime.utcnow():
        p.status = PairingStatus.EXPIRED
        db.commit()
        _publish(p, "pairing_expired")
        if p.created_by_user_id:
            notification_service.notify(db, p.organization_id, p.created_by_user_id, "PAIRING_EXPIRED",
                                        "QR pairing expired", "A QR code expired before verification completed.",
                                        link="/verify/qr")
            db.commit()
        return True
    return False


# ── operator side ───────────────────────────────────────────────────────────

def create_qr(db: Session, actor: Actor, target_user_id: str, camera_id: Optional[str], ip: str) -> tuple[CapturePairing, str]:
    target = db.query(User).filter(User.id == target_user_id).first()
    if target is None or not target.is_active or membership_role(db, actor.org_id, target_user_id) is None:
        raise ApiError(404, "TARGET_NOT_FOUND", "Target user not found or disabled")
    from backend.services.enrolment_service import get_enrolment
    try:
        capture_service.check_verifiable(get_enrolment(db, actor.org_id, target.id))
    except capture_service.CaptureError as exc:
        raise ApiError(exc.status, exc.code, exc.message) from exc
    if camera_id:
        capture_service._camera(db, actor, camera_id)
    now = datetime.utcnow()
    for old in (db.query(CapturePairing)
                .filter(CapturePairing.organization_id == actor.org_id, CapturePairing.created_by_user_id == actor.id,
                        CapturePairing.kind == PairingKind.QR,
                        CapturePairing.status.in_([PairingStatus.WAITING, PairingStatus.AWAITING_CONFIRMATION]))):
        old.status, old.cancelled_reason = PairingStatus.CANCELLED, "SUPERSEDED"
    policy = get_active_policy(db, actor.org_id)
    raw = crypto.token(24)
    p = CapturePairing(organization_id=actor.org_id, kind=PairingKind.QR, status=PairingStatus.WAITING,
                       token_hash=crypto.keyed_hash(raw, "pair"), target_user_id=target.id,
                       created_by_user_id=actor.id, camera_id=camera_id, created_at=now,
                       expires_at=now + timedelta(seconds=int(policy.pairing_ttl_seconds)))
    db.add(p)
    db.commit()
    audit(db, actor.id, "PAIRING_CREATED", "pairing", p.id, {"target_user_id": target.id}, ip, org_id=actor.org_id)
    return p, raw


def create_hosted(db: Session, org_id: str, client_id: str, target: User, external_ref: Optional[str],
                  ttl_seconds: int) -> tuple[CapturePairing, str]:
    """Integration API: a link the client delivers to its end user. No operator confirmation step."""
    now = datetime.utcnow()
    raw = crypto.token(24)
    p = CapturePairing(organization_id=org_id, kind=PairingKind.HOSTED, status=PairingStatus.WAITING,
                       token_hash=crypto.keyed_hash(raw, "pair"), target_user_id=target.id,
                       integration_client_id=client_id, external_ref=external_ref, created_at=now,
                       expires_at=now + timedelta(seconds=ttl_seconds))
    db.add(p)
    db.flush()
    return p, raw


def load_for_operator(db: Session, actor: Actor, pairing_id: str) -> CapturePairing:
    p = (db.query(CapturePairing)
         .filter(CapturePairing.id == pairing_id, CapturePairing.organization_id == actor.org_id).first())
    if p is None or (p.created_by_user_id != actor.id and not actor.can(Permission.HISTORY_READ_ALL)):
        raise ApiError(404, "NOT_FOUND", "Pairing not found")
    _expire_if_needed(db, p)
    return p


def confirm(db: Session, actor: Actor, pairing_id: str, code: str, ip: str) -> CapturePairing:
    p = load_for_operator(db, actor, pairing_id)
    if p.created_by_user_id != actor.id:
        raise ApiError(403, "NOT_CREATOR", "Only the operator who created this pairing can confirm it")
    if p.status != PairingStatus.AWAITING_CONFIRMATION:
        raise ApiError(409, "NOT_AWAITING_CONFIRMATION", f"Pairing is {p.status.lower()}")
    if crypto.constant_time_equals(crypto.keyed_hash((code or "").strip(), "paircode"), p.confirm_code_hash or ""):
        p.status = PairingStatus.READY
        p.confirmed_at = datetime.utcnow()
        p.expires_at = datetime.utcnow() + CAPTURE_WINDOW
        db.commit()
        audit(db, actor.id, "PAIRING_CONFIRMED", "pairing", p.id, None, ip, org_id=p.organization_id)
        _publish(p, "pairing_confirmed")
        return p
    p.confirm_attempts += 1
    if p.confirm_attempts >= MAX_CONFIRM_ATTEMPTS:
        p.status, p.cancelled_reason = PairingStatus.CANCELLED, "CONFIRMATION_FAILED"
        db.commit()
        log_security_event(db, "PAIRING_CONFIRMATION_FAILED", "Pairing cancelled after wrong confirmation codes",
                           "HIGH", user_id=actor.id, ip_address=ip, org_id=p.organization_id,
                           metadata={"pairing_id": p.id})
        alert_service.raise_alert(db, "PAIRING_ABUSE", "HIGH", "pairing", p.id,
                                  "A paired phone's code did not match the operator's entry three times — "
                                  "the QR may have been claimed by someone else", org_id=p.organization_id)
        _publish(p, "pairing_cancelled", {"reason": "CONFIRMATION_FAILED"})
        raise ApiError(409, "PAIRING_CANCELLED", "Too many wrong codes — the pairing was cancelled. Create a new QR.")
    db.commit()
    raise ApiError(400, "WRONG_CODE", f"The code does not match. {MAX_CONFIRM_ATTEMPTS - p.confirm_attempts} attempt(s) left.")


def cancel(db: Session, p: CapturePairing, reason: str, actor_id: Optional[str], ip: str) -> CapturePairing:
    if p.status not in OPEN_PAIRING_STATUSES:
        return p
    if p.session_id:
        s = db.query(VerificationSession).filter(VerificationSession.id == p.session_id).first()
        if s is not None and s.is_active:
            capture_service._close(db, s, capture_service.OutcomeEnum.SESSION_CANCELLED, "PAIRING_CANCELLED",
                                   datetime.utcnow())
    p.status, p.cancelled_reason = PairingStatus.CANCELLED, reason
    db.commit()
    audit(db, actor_id, "PAIRING_CANCELLED", "pairing", p.id, {"reason": reason}, ip, org_id=p.organization_id)
    _publish(p, "pairing_cancelled", {"reason": reason})
    return p


# ── phone side ──────────────────────────────────────────────────────────────

def _by_token(db: Session, raw: str) -> CapturePairing:
    if not raw or len(raw) > 64:
        raise ApiError(404, "PAIRING_NOT_FOUND", "This verification link is not valid")
    p = db.query(CapturePairing).filter(CapturePairing.token_hash == crypto.keyed_hash(raw, "pair")).first()
    if p is None:
        raise ApiError(404, "PAIRING_NOT_FOUND", "This verification link is not valid")
    return p


def _gone(p: CapturePairing) -> ApiError:
    if p.status == PairingStatus.EXPIRED:
        return ApiError(410, "PAIRING_EXPIRED", "This QR code has expired. Ask for a new one.")
    if p.status == PairingStatus.CANCELLED:
        return ApiError(410, "PAIRING_CANCELLED", "This verification was cancelled.")
    return ApiError(410, "PAIRING_USED", "This QR code has already been used.")


def claim_info(db: Session, raw: str) -> dict[str, Any]:
    p = _by_token(db, raw)
    _expire_if_needed(db, p)
    if p.status != PairingStatus.WAITING:
        raise _gone(p)
    org = db.query(Organization).filter(Organization.id == p.organization_id).first()
    target = db.query(User).filter(User.id == p.target_user_id).first()
    operator = db.query(User).filter(User.id == p.created_by_user_id).first() if p.created_by_user_id else None
    return {"kind": p.kind, "organization": org.name if org else None,
            "requested_by": operator.full_name if operator else None,
            "subject": masked_name(target.full_name if target else ""), "expires_at": p.expires_at,
            "needs_operator_confirmation": p.kind == PairingKind.QR}


def _rate_limit_claims(db: Session, ip: str) -> None:
    since = datetime.utcnow() - timedelta(minutes=10)
    n = (db.query(LoginAttempt).filter(LoginAttempt.identifier == ip, LoginAttempt.identifier_type == "pair",
                                       LoginAttempt.attempted_at >= since).count())
    if n >= CLAIMS_PER_IP_PER_10MIN:
        raise ApiError(429, "TOO_MANY_ATTEMPTS", "Too many attempts from this network. Try again later.",
                       headers={"Retry-After": "600"})


def claim(db: Session, raw: str, ip: str, user_agent: str) -> tuple[CapturePairing, str, Optional[str]]:
    _rate_limit_claims(db, ip)
    p = _by_token(db, raw)
    db.add(LoginAttempt(identifier=ip, identifier_type="pair", success=True, ip_address=ip))
    db.commit()
    _expire_if_needed(db, p)
    secret = crypto.token(32)
    code = crypto.numeric_code(6) if p.kind == PairingKind.QR else None
    new_status = PairingStatus.AWAITING_CONFIRMATION if p.kind == PairingKind.QR else PairingStatus.READY
    now = datetime.utcnow()
    values: dict[str, Any] = {"status": new_status, "secret_hash": crypto.keyed_hash(secret, "pairsecret"),
                              "claimed_at": now, "claim_ip": ip[:45], "claim_user_agent": (user_agent or "")[:200]}
    if code:
        values["confirm_code_hash"] = crypto.keyed_hash(code, "paircode")
    else:
        values["confirmed_at"] = now
        values["expires_at"] = now + CAPTURE_WINDOW
    n = (db.query(CapturePairing)
         .filter(CapturePairing.id == p.id, CapturePairing.status == PairingStatus.WAITING,
                 CapturePairing.expires_at >= now)
         .update(values, synchronize_session=False))
    db.commit()
    db.refresh(p)
    if n != 1:
        log_security_event(db, "PAIRING_REUSE", f"Attempt to claim pairing {p.id[:8]} in state {p.status}", "WARNING",
                           ip_address=ip, org_id=p.organization_id, metadata={"pairing_id": p.id})
        raise _gone(p)
    log_security_event(db, "PAIRING_CLAIMED", f"Pairing {p.id[:8]} claimed by a device", "INFO", ip_address=ip,
                       org_id=p.organization_id, metadata={"pairing_id": p.id})
    _publish(p, "pairing_claimed", {"device": _device_label(user_agent)})
    return p, secret, code


def _device_label(ua: str) -> str:
    ua = ua or ""
    for key, label in (("iPhone", "iPhone"), ("iPad", "iPad"), ("Android", "Android device"), ("Windows", "Windows PC"),
                       ("Macintosh", "Mac"), ("Linux", "Linux device")):
        if key in ua:
            return label
    return "a device"


def by_secret(db: Session, secret: str) -> CapturePairing:
    if not secret or len(secret) > 128:
        raise ApiError(401, "INVALID_PAIRING", "This device is not paired")
    p = db.query(CapturePairing).filter(CapturePairing.secret_hash == crypto.keyed_hash(secret, "pairsecret")).first()
    if p is None:
        raise ApiError(401, "INVALID_PAIRING", "This device is not paired")
    return p


def actor_for_secret(db: Session, secret: str) -> Actor:
    p = by_secret(db, secret)
    if p.status not in (PairingStatus.READY, PairingStatus.IN_PROGRESS, PairingStatus.COMPLETED):
        raise ApiError(403, "PAIRING_NOT_READY", f"Pairing is {p.status.lower()}")
    return Actor(kind="PAIRING", org_id=p.organization_id, ref=p.id, label="paired device")


def start_session(db: Session, secret: str, ip: str) -> dict[str, Any]:
    p = by_secret(db, secret)
    if p.status == PairingStatus.AWAITING_CONFIRMATION:
        raise ApiError(409, "AWAITING_CONFIRMATION", "Waiting for the operator to confirm the code")
    if p.status not in (PairingStatus.READY, PairingStatus.IN_PROGRESS):
        raise _gone(p) if p.status != PairingStatus.COMPLETED else ApiError(409, "PAIRING_COMPLETED",
                                                                             "This verification has already finished")
    if p.expires_at < datetime.utcnow() and p.status == PairingStatus.READY:
        p.status = PairingStatus.EXPIRED
        db.commit()
        _publish(p, "pairing_expired")
        raise _gone(p)
    policy = get_active_policy(db, p.organization_id)
    if p.sessions_started >= int(policy.pairing_max_sessions):
        raise ApiError(429, "TOO_MANY_ATTEMPTS", "No attempts left for this verification")
    if p.session_id:
        running = db.query(VerificationSession).filter(VerificationSession.id == p.session_id).first()
        if running is not None and running.is_active:
            raise ApiError(409, "SESSION_RUNNING", "A capture for this pairing is already running")
    actor = Actor(kind="PAIRING", org_id=p.organization_id, ref=p.id, label="paired device")
    target = db.query(User).filter(User.id == p.target_user_id).first()
    created = capture_service.create_session(
        db, actor, SessionPurpose.VERIFY, target, ip, channel="MOBILE" if p.kind == PairingKind.QR else "HOSTED",
        camera_id=p.camera_id, pairing_id=p.id, created_by_user_id=p.created_by_user_id)
    p.status = PairingStatus.IN_PROGRESS
    p.session_id = created.session.id
    p.sessions_started += 1
    db.commit()
    _publish(p, "pairing_session_started", {"session_id": created.session.id})
    view = capture_service.session_view(db, created.session, actor, state=created.state)
    view["frame_token"] = created.frame_token
    return view


def on_session_closed(db: Session, s: VerificationSession, _decision=None) -> None:
    """Terminal hook: propagate the session result to its pairing (exactly one final decision)."""
    if not s.pairing_id:
        return
    p = db.query(CapturePairing).filter(CapturePairing.id == s.pairing_id).first()
    if p is None or p.session_id != s.id or p.status != PairingStatus.IN_PROGRESS:
        return
    policy = get_active_policy(db, p.organization_id)
    final = s.final_outcome in FINAL_OUTCOMES or p.sessions_started >= int(policy.pairing_max_sessions)
    if s.reason_code == "PAIRING_CANCELLED":
        return
    if final:
        n = (db.query(CapturePairing)
             .filter(CapturePairing.id == p.id, CapturePairing.status == PairingStatus.IN_PROGRESS)
             .update({"status": PairingStatus.COMPLETED, "result_outcome": s.final_outcome,
                      "completed_at": datetime.utcnow()}, synchronize_session=False))
        if n != 1:
            return
        db.refresh(p)
        if p.created_by_user_id and p.kind == PairingKind.QR:
            kind = {"VERIFIED": "VERIFICATION_COMPLETED", "LIVENESS_FAILED": "LIVENESS_FAILED"}.get(
                s.final_outcome, "VERIFICATION_REJECTED")
            notification_service.notify(db, p.organization_id, p.created_by_user_id, kind,
                                        f"Mobile verification: {s.final_outcome.replace('_', ' ').lower()}",
                                        "The paired phone finished its verification.",
                                        link=f"/history/{s.id}")
    else:
        p.status = PairingStatus.READY           # retry allowed (e.g. expired or cancelled on the phone)
    _publish(p, "pairing_result" if final else "pairing_session_ended",
             {"session_id": s.id, "outcome": s.final_outcome, "review_status": s.review_status})


def publish_progress(s: VerificationSession, frame: dict[str, Any], challenge: Optional[dict]) -> None:
    """Per-frame progress for the desktop that created a paired session (no images, no scores)."""
    if not s.pairing_id or not s.created_by_user_id:
        return
    bus.publish("pairing", "pairing_progress", {
        "pairing_id": s.pairing_id, "session_id": s.id, "outcome": frame.get("outcome"),
        "face_count": frame.get("face_count"), "message": frame.get("message"),
        "guidance": frame.get("guidance"), "terminal": frame.get("terminal"),
        "step": challenge.get("type") if challenge else None,
        "step_index": challenge.get("index") if challenge else None,
        "steps_total": challenge.get("total") if challenge else None,
    }, org_id=s.organization_id, user_id=s.created_by_user_id)


def view(db: Session, p: CapturePairing, include_session: bool = True) -> dict[str, Any]:
    target = db.query(User).filter(User.id == p.target_user_id).first()
    out = {"id": p.id, "kind": p.kind, "status": p.status, "created_at": p.created_at, "expires_at": p.expires_at,
           "claimed_at": p.claimed_at, "confirmed_at": p.confirmed_at, "completed_at": p.completed_at,
           "target_user_id": p.target_user_id, "target_name": target.full_name if target else None,
           "camera_id": p.camera_id, "session_id": p.session_id, "sessions_started": p.sessions_started,
           "result_outcome": p.result_outcome, "cancelled_reason": p.cancelled_reason,
           "device": _device_label(p.claim_user_agent or "") if p.claimed_at else None,
           "external_ref": p.external_ref}
    if include_session and p.session_id:
        s = db.query(VerificationSession).filter(VerificationSession.id == p.session_id).first()
        if s is not None:
            out["session"] = {"status": "ACTIVE" if s.is_active else "CLOSED", "final_outcome": s.final_outcome,
                              "reason_code": s.reason_code, "frames": s.frame_count,
                              "review_status": s.review_status}
    return out


def phone_view(db: Session, p: CapturePairing) -> dict[str, Any]:
    """What the paired phone may see: no operator identities, no scores."""
    target = db.query(User).filter(User.id == p.target_user_id).first()
    org = db.query(Organization).filter(Organization.id == p.organization_id).first()
    out = {"status": p.status, "kind": p.kind, "organization": org.name if org else None,
           "subject": masked_name(target.full_name if target else ""), "expires_at": p.expires_at,
           "result_outcome": p.result_outcome, "session_id": p.session_id, "attempts_used": p.sessions_started,
           "cancelled_reason": p.cancelled_reason}
    if p.session_id:
        s = db.query(VerificationSession).filter(VerificationSession.id == p.session_id).first()
        if s is not None:
            out["session_status"] = "ACTIVE" if s.is_active else "CLOSED"
            out["session_outcome"] = s.final_outcome
            out["review_status"] = s.review_status
    return out


def expire_task(db: Session, now: datetime, _policies: dict) -> dict[str, int]:
    n = 0
    for p in (db.query(CapturePairing)
              .filter(CapturePairing.status.in_([PairingStatus.WAITING, PairingStatus.AWAITING_CONFIRMATION,
                                                 PairingStatus.READY]),
                      CapturePairing.expires_at < now).all()):
        if p.status == PairingStatus.READY:
            p.status = PairingStatus.EXPIRED
            db.commit()
            _publish(p, "pairing_expired")
            n += 1
        elif _expire_if_needed(db, p):
            n += 1
    return {"pairings_expired": n}
