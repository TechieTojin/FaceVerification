"""Review queue: list, detail, assign, decide, comment."""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.core.deps import Actor, client_ip, require
from backend.core.permissions import Permission, permissions_for
from backend.database import get_db
from backend.models.organization import Membership, MembershipStatus
from backend.models.reviews import OPEN_REVIEW_STATUSES, VerificationReview
from backend.models.user import User
from backend.services import review_service

router = APIRouter(prefix="/api/v1/reviews", tags=["reviews"])


class AssignBody(BaseModel):
    assignee_user_id: str = Field(..., max_length=36)


class DecideBody(BaseModel):
    decision: Literal["APPROVE", "REJECT", "REQUEST_RETRY"]
    note: str = Field(..., max_length=2000)
    version: int = Field(..., ge=1)


class CommentBody(BaseModel):
    note: str = Field(..., min_length=1, max_length=2000)


@router.get("")
def list_reviews(status: Optional[Literal["OPEN", "PENDING", "IN_REVIEW", "APPROVED", "REJECTED", "RETRY_REQUESTED",
                                          "EXPIRED", "ALL"]] = "OPEN",
                 risk: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None, mine: bool = False,
                 page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
                 db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.REVIEW_READ))):
    q = db.query(VerificationReview).filter(VerificationReview.organization_id == actor.org_id)
    if status == "OPEN":
        q = q.filter(VerificationReview.status.in_(OPEN_REVIEW_STATUSES))
    elif status and status != "ALL":
        q = q.filter(VerificationReview.status == status)
    if risk:
        q = q.filter(VerificationReview.risk_level == risk)
    if mine:
        q = q.filter(VerificationReview.assigned_to_user_id == actor.id)
    total = q.count()
    rows = q.order_by(VerificationReview.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    counts = dict(db.query(VerificationReview.status, func.count())
                  .filter(VerificationReview.organization_id == actor.org_id)
                  .group_by(VerificationReview.status).all())
    return {"total": total, "page": page, "page_size": page_size, "counts": counts,
            "reviews": [review_service.view(db, r) for r in rows]}


@router.get("/reviewers")
def reviewers(db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.REVIEW_DECIDE))):
    rows = (db.query(User, Membership).join(Membership, Membership.user_id == User.id)
            .filter(Membership.organization_id == actor.org_id, Membership.status == MembershipStatus.ACTIVE,
                    User.is_active == True).all())  # noqa: E712
    return {"reviewers": [{"id": u.id, "full_name": u.full_name, "email": u.email} for u, m in rows
                          if Permission.REVIEW_DECIDE in permissions_for(m.role)]}


@router.get("/{review_id}")
def get_review(review_id: str, db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.REVIEW_READ))):
    return review_service.view(db, review_service.load(db, actor, review_id), detail=True)


@router.post("/{review_id}/assign")
def assign(review_id: str, body: AssignBody, request: Request, db: Session = Depends(get_db),
           actor: Actor = Depends(require(Permission.REVIEW_DECIDE))):
    r = review_service.assign(db, actor, review_id, body.assignee_user_id, client_ip(request))
    return review_service.view(db, r, detail=True)


@router.post("/{review_id}/decide")
def decide(review_id: str, body: DecideBody, request: Request, db: Session = Depends(get_db),
           actor: Actor = Depends(require(Permission.REVIEW_DECIDE))):
    r = review_service.decide(db, actor, review_id, body.decision, body.note, body.version, client_ip(request))
    return review_service.view(db, r, detail=True)


@router.post("/{review_id}/comments")
def add_comment(review_id: str, body: CommentBody, db: Session = Depends(get_db),
                actor: Actor = Depends(require(Permission.REVIEW_READ))):
    return review_service.view(db, review_service.comment(db, actor, review_id, body.note), detail=True)
