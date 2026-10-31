"""Biometric enrolment, consent and approval endpoints."""
from __future__ import annotations

import json
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.deps import Actor, client_ip, get_actor, membership_role, require
from backend.core.permissions import Permission
from backend.database import get_db
from backend.models.biometric import ConsentRecord, EnrolmentRequest
from backend.models.organization import Membership
from backend.models.user import User
from backend.models.verification import SessionPurpose
from backend.services import capture_service, enrolment_service
from backend.services.policy_service import get_active_policy

router = APIRouter(prefix="/api/v1/enrolments", tags=["enrolments"])


class ConsentBody(BaseModel):
    subject_password: str = Field(..., min_length=1, max_length=256)
    accept: bool


class ReasonBody(BaseModel):
    reason: str = Field("", max_length=500)


class ReviewBody(BaseModel):
    note: str = Field("", max_length=1000)


def _target(db: Session, actor: Actor, user_id: str) -> User:
    """A member (any membership status) of the actor's organisation, else 404."""
    u = (db.query(User).join(Membership, Membership.user_id == User.id)
         .filter(User.id == user_id, Membership.organization_id == actor.org_id).first())
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return u


def _raise(exc: enrolment_service.EnrolmentDenied):
    raise exc  # rendered as {"detail", "code"} by the handler in main.py


def request_view(db: Session, r: EnrolmentRequest, actor: Actor) -> dict:
    names = {u.id: u for u in db.query(User).filter(User.id.in_(
        [x for x in (r.user_id, r.requested_by_user_id, r.reviewed_by_user_id) if x])).all()}
    v = {"id": r.id, "user_id": r.user_id, "user_name": getattr(names.get(r.user_id), "full_name", None),
         "user_email": getattr(names.get(r.user_id), "email", None), "kind": r.kind, "status": r.status,
         "approval_reasons": json.loads(r.approval_reasons or "[]"), "sample_count": r.sample_count,
         "quality_score": r.quality_score, "consistency_score": r.consistency_score, "model_name": r.model_name,
         "requested_by": getattr(names.get(r.requested_by_user_id), "email", None),
         "requested_by_user_id": r.requested_by_user_id, "created_at": r.created_at,
         "reviewed_by": getattr(names.get(r.reviewed_by_user_id), "email", None), "reviewed_at": r.reviewed_at,
         "review_note": r.review_note, "session_id": r.session_id}
    if actor.can(Permission.ENROL_APPROVE):
        v["duplicate_user_ids"] = json.loads(r.duplicate_user_ids) if r.duplicate_user_ids else []
    return v


@router.get("/consent-text")
def consent_text(_: Actor = Depends(get_actor)):
    return {"version": enrolment_service.CONSENT_VERSION, "text": enrolment_service.CONSENT_TEXT}


@router.get("/requests")
def list_requests(status: Optional[Literal["PENDING", "APPROVED", "REJECTED", "AUTO_APPROVED", "CANCELLED"]] = None,
                  user_id: Optional[str] = None, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
                  db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.ENROL_READ))):
    q = db.query(EnrolmentRequest).filter(EnrolmentRequest.organization_id == actor.org_id)
    if not actor.can(Permission.ENROL_APPROVE):
        q = q.filter(EnrolmentRequest.requested_by_user_id == actor.id)
    if status:
        q = q.filter(EnrolmentRequest.status == status)
    if user_id:
        q = q.filter(EnrolmentRequest.user_id == user_id)
    total = q.count()
    rows = q.order_by(EnrolmentRequest.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size, "requests": [request_view(db, r, actor) for r in rows]}


@router.post("/requests/{request_id}/approve")
def approve(request_id: str, body: ReviewBody, request: Request, db: Session = Depends(get_db),
            actor: Actor = Depends(get_actor)):
    try:
        r = enrolment_service.review_request(db, actor, request_id, True, body.note, get_active_policy(db, actor.org_id), client_ip(request))
    except enrolment_service.EnrolmentDenied as exc:
        _raise(exc)
    return request_view(db, r, actor)


@router.post("/requests/{request_id}/reject")
def reject(request_id: str, body: ReviewBody, request: Request, db: Session = Depends(get_db),
           actor: Actor = Depends(get_actor)):
    try:
        r = enrolment_service.review_request(db, actor, request_id, False, body.note, get_active_policy(db, actor.org_id), client_ip(request))
    except enrolment_service.EnrolmentDenied as exc:
        _raise(exc)
    return request_view(db, r, actor)


@router.get("/{user_id}")
def status(user_id: str, db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    if actor.id != user_id and not actor.can(Permission.ENROL_READ):
        raise HTTPException(status_code=403, detail="You do not have permission to view this enrolment")
    target = _target(db, actor, user_id)
    enr = enrolment_service.get_enrolment(db, actor.org_id, user_id)
    consents = (db.query(ConsentRecord)
                .filter(ConsentRecord.organization_id == actor.org_id, ConsentRecord.user_id == user_id)
                .order_by(ConsentRecord.recorded_at.desc()).limit(20).all())
    return {
        "user_id": user_id, "user_name": target.full_name, "user_email": target.email, "user_role": getattr(membership_role(db, actor.org_id, user_id), "value", None),
        **enrolment_service.enrolment_status_view(enr),
        "consent_history": [{"granted": c.consent_given, "recorded_at": c.recorded_at, "version": c.consent_version,
                             "purpose": c.purpose, "method": c.method} for c in consents],
    }


@router.post("/{user_id}/consent")
def grant_consent(user_id: str, body: ConsentBody, request: Request, db: Session = Depends(get_db),
                  actor: Actor = Depends(get_actor)):
    if not body.accept:
        raise HTTPException(status_code=422, detail="Consent must be explicitly accepted")
    target = _target(db, actor, user_id)
    try:
        enr = enrolment_service.record_consent(db, actor, target, body.subject_password, client_ip(request))
    except enrolment_service.EnrolmentDenied as exc:
        _raise(exc)
    return enrolment_service.enrolment_status_view(enr)


@router.post("/{user_id}/consent/withdraw")
def withdraw_consent(user_id: str, body: ReasonBody, request: Request, db: Session = Depends(get_db),
                     actor: Actor = Depends(get_actor)):
    target = _target(db, actor, user_id)
    try:
        enr = enrolment_service.withdraw_consent(db, actor, target, client_ip(request), body.reason)
    except enrolment_service.EnrolmentDenied as exc:
        _raise(exc)
    return enrolment_service.enrolment_status_view(enr)


@router.post("/{user_id}/sessions", status_code=201)
def start_capture(user_id: str, request: Request, db: Session = Depends(get_db),
                  actor: Actor = Depends(get_actor)):
    """Start an enrolment capture session (liveness challenge + multi-sample capture)."""
    target = _target(db, actor, user_id)
    created = capture_service.create_session(db, actor, SessionPurpose.ENROL, target, client_ip(request))
    view = capture_service.session_view(db, created.session, actor, state=created.state)
    view["frame_token"] = created.frame_token
    return view


@router.delete("/{user_id}")
def revoke(user_id: str, body: ReasonBody, request: Request, db: Session = Depends(get_db),
           actor: Actor = Depends(get_actor)):
    target = _target(db, actor, user_id)
    try:
        enr = enrolment_service.revoke(db, actor, target, body.reason, client_ip(request))
    except enrolment_service.EnrolmentDenied as exc:
        _raise(exc)
    return enrolment_service.enrolment_status_view(enr)


@router.post("/{user_id}/request-reenrolment")
def request_reenrolment(user_id: str, body: ReasonBody, request: Request, db: Session = Depends(get_db),
                        actor: Actor = Depends(get_actor)):
    target = _target(db, actor, user_id)
    try:
        enr = enrolment_service.request_reenrolment(db, actor, target, body.reason, client_ip(request))
    except enrolment_service.EnrolmentDenied as exc:
        _raise(exc)
    return enrolment_service.enrolment_status_view(enr)
