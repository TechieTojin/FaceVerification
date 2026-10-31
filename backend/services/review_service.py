"""Manual review of uncertain or high-risk verification results.

A review is an *administrative disposition* layered on top of a session:
  * the technical outcome (VERIFIED / NOT_MATCHED / LIVENESS_FAILED …) is never changed;
  * a case whose liveness failed or whose passive check labelled the capture a spoof
    can be rejected or sent back for a retry, but never approved;
  * reviews never apply to sign-in face checks, so a reviewer cannot issue or
    complete an authentication session;
  * decisions are atomic (status + version guard): two reviewers deciding at the
    same time produce exactly one decision;
  * the reviewer cannot be the operator who ran the session or its subject.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.core.deps import Actor, membership_role
from backend.core.errors import ApiError
from backend.core.events import bus
from backend.core.permissions import Permission, permissions_for
from backend.models.documents import DocumentCheck
from backend.models.reviews import OPEN_REVIEW_STATUSES, ReviewEvent, ReviewStatus, VerificationReview
from backend.models.user import User
from backend.models.verification import VerificationEvent, VerificationSession
from backend.services import notification_service
from backend.services.audit_service import audit, log_security_event
from backend.services.policy_service import Policy

REASON_TEXT = {
    "BORDERLINE_MATCH": "Match distance is within the borderline band around the threshold",
    "LOW_LIVENESS_CONFIDENCE": "Passive anti-spoofing passed with low confidence",
    "SPOOF_DETECTED": "Passive anti-spoofing classified the capture as a presentation attack",
    "DOCUMENT_AUTHENTICITY_UNVERIFIED": "Face matched the document portrait, but document authenticity was not checked",
    "REMOTE_CAPTURE_FAILED_LIVENESS": "A remote (phone or hosted) capture failed liveness",
}
DECISIONS = {"APPROVE": ReviewStatus.APPROVED, "REJECT": ReviewStatus.REJECTED,
             "REQUEST_RETRY": ReviewStatus.RETRY_REQUESTED}


def _passive(db: Session, s: VerificationSession) -> tuple[Optional[float], bool]:
    rows = (db.query(VerificationEvent.passive_label, VerificationEvent.passive_confidence)
            .filter(VerificationEvent.session_id == s.id, VerificationEvent.passive_label.isnot(None)).all())
    spoof = any(label == "SPOOF" for label, _ in rows)
    real = [c for label, c in rows if label == "REAL" and c is not None]
    return (min(real) if real else None), spoof


def reasons_for(db: Session, s: VerificationSession, policy: Policy) -> tuple[list[str], bool]:
    reasons: list[str] = []
    min_real, spoof = _passive(db, s)
    if s.final_outcome in ("VERIFIED", "NOT_MATCHED") and s.final_distance is not None and s.threshold is not None:
        if abs(s.final_distance - s.threshold) <= float(policy.review_distance_margin):
            reasons.append("BORDERLINE_MATCH")
    if s.final_outcome == "VERIFIED" and min_real is not None and min_real < float(policy.review_passive_confidence_floor):
        reasons.append("LOW_LIVENESS_CONFIDENCE")
    if s.final_outcome == "LIVENESS_FAILED" and spoof:
        reasons.append("SPOOF_DETECTED")
    elif s.final_outcome == "LIVENESS_FAILED" and s.channel in ("MOBILE", "HOSTED"):
        reasons.append("REMOTE_CAPTURE_FAILED_LIVENESS")
    if s.purpose == "DOCUMENT" and s.document_check_id and s.final_outcome == "VERIFIED":
        doc = s.document_check_id and _doc(db, s.document_check_id)
        if doc is not None and doc.authenticity_status != "PASSED":
            reasons.append("DOCUMENT_AUTHENTICITY_UNVERIFIED")
    return reasons, spoof


def _doc(db: Session, doc_id: str) -> Optional[DocumentCheck]:
    return db.query(DocumentCheck).filter(DocumentCheck.id == doc_id).first()


def on_session_closed(db: Session, s: VerificationSession, _decision=None) -> None:
    """Terminal hook: open a review case when policy and the result call for one."""
    if s.owner_kind == "LOGIN" or s.purpose not in ("VERIFY", "DOCUMENT"):
        return
    if s.final_outcome not in ("VERIFIED", "NOT_MATCHED", "LIVENESS_FAILED"):
        return
    policy = Policy.from_json(s.policy_snapshot)
    if not policy.review_enabled:
        return
    if db.query(VerificationReview).filter(VerificationReview.session_id == s.id).first():
        return
    reasons, spoof = reasons_for(db, s, policy)
    if not reasons:
        return
    risk = "HIGH" if spoof or "REMOTE_CAPTURE_FAILED_LIVENESS" in reasons else (
        "MEDIUM" if "BORDERLINE_MATCH" in reasons or "LOW_LIVENESS_CONFIDENCE" in reasons else "LOW")
    now = datetime.utcnow()
    r = VerificationReview(organization_id=s.organization_id, session_id=s.id, document_check_id=s.document_check_id,
                           status=ReviewStatus.PENDING, risk_level=risk, reasons_json=json.dumps(reasons),
                           technical_outcome=s.final_outcome, spoof_confirmed=spoof,
                           subject_user_id=s.target_user_id, created_at=now,
                           expires_at=now + timedelta(hours=int(policy.review_expiry_hours)))
    db.add(r)
    db.flush()
    db.add(ReviewEvent(review_id=r.id, action="CREATED", note=", ".join(reasons), created_at=now))
    s.review_status = ReviewStatus.PENDING
    db.query(VerificationSession).filter(VerificationSession.id == s.id).update(
        {"review_status": ReviewStatus.PENDING}, synchronize_session=False)
    notification_service.notify_permission(db, s.organization_id, "REVIEW_ASSIGNED", "New verification to review",
                                           REASON_TEXT[reasons[0]], link=f"/reviews/{r.id}", severity="WARNING",
                                           permission=Permission.REVIEW_DECIDE.value,
                                           exclude_user_id=s.created_by_user_id)
    bus.publish("reviews", "review_created", {"id": r.id, "risk": risk, "reasons": reasons},
                org_id=s.organization_id)


def load(db: Session, actor: Actor, review_id: str) -> VerificationReview:
    r = (db.query(VerificationReview)
         .filter(VerificationReview.id == review_id, VerificationReview.organization_id == actor.org_id).first())
    if r is None:
        raise ApiError(404, "NOT_FOUND", "Review not found")
    _expire(db, r)
    return r


def _expire(db: Session, r: VerificationReview) -> bool:
    if r.status in OPEN_REVIEW_STATUSES and r.expires_at < datetime.utcnow():
        r.status = ReviewStatus.EXPIRED
        r.version += 1
        db.add(ReviewEvent(review_id=r.id, action="EXPIRED", created_at=datetime.utcnow()))
        if r.session_id:
            db.query(VerificationSession).filter(VerificationSession.id == r.session_id).update(
                {"review_status": ReviewStatus.EXPIRED}, synchronize_session=False)
        db.commit()
        return True
    return False


def _is_reviewer(db: Session, org_id: str, user_id: str) -> bool:
    role = membership_role(db, org_id, user_id)
    return role is not None and Permission.REVIEW_DECIDE in permissions_for(role)


def _session(db: Session, r: VerificationReview) -> Optional[VerificationSession]:
    return db.query(VerificationSession).filter(VerificationSession.id == r.session_id).first() if r.session_id else None


def assign(db: Session, actor: Actor, review_id: str, assignee_id: str, ip: str) -> VerificationReview:
    r = load(db, actor, review_id)
    if r.status not in OPEN_REVIEW_STATUSES:
        raise ApiError(409, "NOT_OPEN", f"Review is {r.status.lower()}")
    if not _is_reviewer(db, actor.org_id, assignee_id):
        raise ApiError(422, "NOT_A_REVIEWER", "The assignee cannot review verifications in this organisation")
    s = _session(db, r)
    if s is not None and assignee_id in (s.created_by_user_id, s.target_user_id):
        raise ApiError(409, "CONFLICT_OF_INTEREST", "The operator or the subject of a session cannot review it")
    version = r.version
    n = (db.query(VerificationReview)
         .filter(VerificationReview.id == r.id, VerificationReview.version == version,
                 VerificationReview.status.in_(OPEN_REVIEW_STATUSES))
         .update({"assigned_to_user_id": assignee_id, "assigned_at": datetime.utcnow(),
                  "status": ReviewStatus.IN_REVIEW, "version": version + 1}, synchronize_session=False))
    if n != 1:
        db.rollback()
        raise ApiError(409, "CONCURRENT_UPDATE", "The review changed meanwhile; reload and retry")
    db.add(ReviewEvent(review_id=r.id, actor_user_id=actor.id, action="ASSIGNED", note=assignee_id))
    if assignee_id != actor.id:
        notification_service.notify(db, actor.org_id, assignee_id, "REVIEW_ASSIGNED", "A review was assigned to you",
                                    REASON_TEXT.get(json.loads(r.reasons_json)[0], ""), link=f"/reviews/{r.id}",
                                    severity="WARNING")
    db.commit()
    db.refresh(r)
    audit(db, actor.id, "REVIEW_ASSIGNED", "review", r.id, {"assignee": assignee_id}, ip, org_id=actor.org_id)
    bus.publish("reviews", "review_updated", {"id": r.id, "status": r.status}, org_id=actor.org_id)
    return r


def decide(db: Session, actor: Actor, review_id: str, decision: str, note: str, version: int, ip: str) -> VerificationReview:
    r = load(db, actor, review_id)
    if decision not in DECISIONS:
        raise ApiError(422, "INVALID_DECISION", "Decision must be APPROVE, REJECT or REQUEST_RETRY")
    if not (note or "").strip():
        raise ApiError(422, "NOTE_REQUIRED", "Explain the decision in a note")
    if r.status not in OPEN_REVIEW_STATUSES:
        raise ApiError(409, "ALREADY_DECIDED", f"Review is already {r.status.lower()}")
    s = _session(db, r)
    if s is not None and actor.id in (s.created_by_user_id, s.target_user_id):
        raise ApiError(409, "CONFLICT_OF_INTEREST", "The operator or the subject of a session cannot review it")
    if r.assigned_to_user_id and r.assigned_to_user_id != actor.id:
        raise ApiError(409, "ASSIGNED_TO_SOMEONE_ELSE", "This review is assigned to another reviewer")
    if decision == "APPROVE" and (r.spoof_confirmed or r.technical_outcome == "LIVENESS_FAILED"):
        log_security_event(db, "REVIEW_OVERRIDE_BLOCKED", f"{actor.email} tried to approve a failed-liveness case",
                           "HIGH", user_id=actor.id, ip_address=ip, org_id=actor.org_id, metadata={"review_id": r.id})
        raise ApiError(409, "CANNOT_APPROVE_LIVENESS_FAILURE",
                       "A case whose liveness check failed cannot be approved. Reject it or request a retry.")
    status = DECISIONS[decision]
    now = datetime.utcnow()
    n = (db.query(VerificationReview)
         .filter(VerificationReview.id == r.id, VerificationReview.version == version,
                 VerificationReview.status.in_(OPEN_REVIEW_STATUSES))
         .update({"status": status, "decided_by_user_id": actor.id, "decided_at": now,
                  "decision_note": note.strip()[:2000], "version": version + 1,
                  "assigned_to_user_id": r.assigned_to_user_id or actor.id}, synchronize_session=False))
    if n != 1:
        db.rollback()
        raise ApiError(409, "CONCURRENT_UPDATE", "Another reviewer decided this case first; reload to see the result")
    if s is not None:
        db.query(VerificationSession).filter(VerificationSession.id == s.id).update(
            {"review_status": status}, synchronize_session=False)
    db.add(ReviewEvent(review_id=r.id, actor_user_id=actor.id, action=decision, note=note.strip()[:2000], created_at=now))
    if s is not None and s.created_by_user_id and s.created_by_user_id != actor.id:
        notification_service.notify(db, actor.org_id, s.created_by_user_id, "REVIEW_COMPLETED",
                                    f"Review {status.lower().replace('_', ' ')}",
                                    "A reviewer decided a verification you ran.", link=f"/history/{s.id}")
    db.commit()
    db.refresh(r)
    audit(db, actor.id, f"REVIEW_{status}", "review", r.id,
          {"session_id": r.session_id, "technical_outcome": r.technical_outcome, "note": note[:300]}, ip,
          org_id=actor.org_id)
    bus.publish("reviews", "review_updated", {"id": r.id, "status": r.status}, org_id=actor.org_id)
    if s is not None:
        from backend.services import integration_service
        integration_service.on_review_decided(db, s, r)
    return r


def comment(db: Session, actor: Actor, review_id: str, note: str) -> VerificationReview:
    r = load(db, actor, review_id)
    if not (note or "").strip():
        raise ApiError(422, "NOTE_REQUIRED", "Comment is empty")
    db.add(ReviewEvent(review_id=r.id, actor_user_id=actor.id, action="COMMENT", note=note.strip()[:2000]))
    db.commit()
    return r


def view(db: Session, r: VerificationReview, detail: bool = False) -> dict[str, Any]:
    ids = {r.assigned_to_user_id, r.decided_by_user_id, r.subject_user_id}
    s = _session(db, r)
    if s is not None:
        ids.add(s.created_by_user_id)
    names = {u.id: u for u in db.query(User).filter(User.id.in_([i for i in ids if i])).all()}
    reasons = json.loads(r.reasons_json)
    out = {"id": r.id, "status": r.status, "risk_level": r.risk_level, "reasons": reasons,
           "reason_text": [REASON_TEXT.get(x, x) for x in reasons], "technical_outcome": r.technical_outcome,
           "spoof_confirmed": r.spoof_confirmed, "can_approve": not (r.spoof_confirmed or r.technical_outcome == "LIVENESS_FAILED"),
           "session_id": r.session_id, "document_check_id": r.document_check_id,
           "subject_user_id": r.subject_user_id,
           "subject_name": getattr(names.get(r.subject_user_id), "full_name", None),
           "assigned_to_user_id": r.assigned_to_user_id,
           "assigned_to": getattr(names.get(r.assigned_to_user_id), "full_name", None),
           "decided_by": getattr(names.get(r.decided_by_user_id), "full_name", None), "decided_at": r.decided_at,
           "decision_note": r.decision_note, "created_at": r.created_at, "expires_at": r.expires_at,
           "version": r.version,
           "channel": s.channel if s else None,
           "operator_user_id": s.created_by_user_id if s else None,
           "operator_name": getattr(names.get(s.created_by_user_id), "full_name", None) if s else None}
    if detail:
        evs = db.query(ReviewEvent).filter(ReviewEvent.review_id == r.id).order_by(ReviewEvent.created_at).all()
        actors = {u.id: u.full_name for u in db.query(User).filter(
            User.id.in_([e.actor_user_id for e in evs if e.actor_user_id])).all()}
        out["history"] = [{"action": e.action, "note": e.note, "at": e.created_at,
                           "actor": actors.get(e.actor_user_id, "System" if not e.actor_user_id else None)}
                          for e in evs]
        if s is not None:
            out["session"] = {"purpose": s.purpose, "final_outcome": s.final_outcome, "reason_code": s.reason_code,
                              "final_distance": s.final_distance, "threshold": s.threshold,
                              "created_at": s.created_at, "completed_at": s.completed_at, "frames": s.frame_count,
                              "attempts_used": s.attempts_used, "policy_version": s.policy_version}
    return out


def expire_task(db: Session, now: datetime, _policies: dict) -> dict[str, int]:
    n = 0
    for r in (db.query(VerificationReview)
              .filter(VerificationReview.status.in_(OPEN_REVIEW_STATUSES), VerificationReview.expires_at < now).all()):
        if _expire(db, r):
            n += 1
    return {"reviews_expired": n}
