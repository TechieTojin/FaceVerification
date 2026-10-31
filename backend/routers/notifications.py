"""Notification centre for the signed-in user (active organisation only)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.core.deps import Actor, get_actor
from backend.core.errors import ApiError
from backend.database import get_db
from backend.models.reviews import Notification
from backend.services import email_service, notification_service

router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])


class PreferenceBody(BaseModel):
    preferences: dict[str, dict[str, bool]]


def _view(n: Notification) -> dict:
    return {"id": n.id, "type": n.type, "severity": n.severity, "title": n.title, "body": n.body, "link": n.link,
            "created_at": n.created_at, "read": n.read_at is not None, "read_at": n.read_at,
            "email_status": n.email_status, "email_attempts": n.email_attempts}


def _mine(db: Session, actor: Actor):
    return db.query(Notification).filter(Notification.user_id == actor.id, Notification.organization_id == actor.org_id)


@router.get("")
def list_notifications(unread_only: bool = False, type: Optional[str] = Query(None, max_length=40),
                       page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
                       db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    q = _mine(db, actor)
    if unread_only:
        q = q.filter(Notification.read_at.is_(None))
    if type:
        q = q.filter(Notification.type == type)
    total = q.count()
    rows = q.order_by(Notification.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size,
            "unread": notification_service.unread_count(db, actor.org_id, actor.id),
            "notifications": [_view(n) for n in rows]}


@router.get("/unread-count")
def unread(db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    return {"unread": notification_service.unread_count(db, actor.org_id, actor.id)}


@router.post("/{notification_id}/read")
def mark_read(notification_id: str, db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    n = _mine(db, actor).filter(Notification.id == notification_id).first()
    if n is None:
        raise ApiError(404, "NOT_FOUND", "Notification not found")
    if n.read_at is None:
        n.read_at = datetime.utcnow()
        db.commit()
    return _view(n)


@router.post("/read-all")
def mark_all_read(db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    n = _mine(db, actor).filter(Notification.read_at.is_(None)).update({"read_at": datetime.utcnow()},
                                                                        synchronize_session=False)
    db.commit()
    return {"marked": n}


@router.get("/preferences")
def get_preferences(db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    return {"preferences": notification_service.preferences_for(db, actor.id),
            "email_available": email_service.available()}


@router.put("/preferences")
def set_preferences(body: PreferenceBody, db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    for type_, pref in body.preferences.items():
        if type_ not in notification_service.TYPES:
            raise ApiError(422, "UNKNOWN_TYPE", f"Unknown notification type {type_}")
        notification_service.set_preference(db, actor.id, type_, bool(pref.get("in_app", True)),
                                            bool(pref.get("email", False)))
    db.commit()
    return {"preferences": notification_service.preferences_for(db, actor.id),
            "email_available": email_service.available()}
