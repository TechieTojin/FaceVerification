"""Security events, alert centre and audit log."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.core.deps import Actor, client_ip, require
from backend.core.events import bus
from backend.core.permissions import Permission
from backend.database import get_db
from backend.models.settings import AuditLog
from backend.models.user import User
from backend.models.verification import SecurityAlert, SecurityEvent
from backend.services.audit_service import audit

router = APIRouter(prefix="/api/v1/security", tags=["security"])


class AlertAction(BaseModel):
    note: str = Field("", max_length=1000)


def _emails(db: Session, ids: set) -> dict:
    ids = {i for i in ids if i}
    return {u.id: u.email for u in db.query(User).filter(User.id.in_(ids)).all()} if ids else {}


@router.get("/events")
def events(severity: Optional[str] = Query(None, max_length=20), event_type: Optional[str] = Query(None, max_length=50),
           page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200),
           db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.SECURITY_READ))):
    q = db.query(SecurityEvent).filter(SecurityEvent.organization_id == actor.org_id)
    if severity:
        q = q.filter(SecurityEvent.severity == severity)
    if event_type:
        q = q.filter(SecurityEvent.event_type == event_type)
    total = q.count()
    rows = q.order_by(SecurityEvent.timestamp.desc()).offset((page - 1) * page_size).limit(page_size).all()
    emails = _emails(db, {r.user_id for r in rows})
    types = [t for (t,) in db.query(SecurityEvent.event_type).filter(SecurityEvent.organization_id == actor.org_id)
             .distinct().order_by(SecurityEvent.event_type).all()]
    return {"total": total, "page": page, "page_size": page_size, "event_types": types, "events": [{
        "id": r.id, "event_type": r.event_type, "description": r.description, "severity": r.severity,
        "timestamp": r.timestamp, "user_id": r.user_id, "user_email": emails.get(r.user_id), "ip_address": r.ip_address,
        "metadata": json.loads(r.metadata_json) if r.metadata_json else None} for r in rows]}


def alert_view(a: SecurityAlert, emails: dict) -> dict:
    return {"id": a.id, "rule": a.rule, "title": a.title, "severity": a.severity, "status": a.status,
            "subject_type": a.subject_type, "subject_id": a.subject_id,
            "subject_label": emails.get(a.subject_id) if a.subject_type == "user" else a.subject_id,
            "description": a.description, "occurrence_count": a.occurrence_count, "first_seen_at": a.first_seen_at,
            "last_seen_at": a.last_seen_at, "acknowledged_by": emails.get(a.acknowledged_by_user_id),
            "acknowledged_at": a.acknowledged_at, "resolved_by": emails.get(a.resolved_by_user_id),
            "resolved_at": a.resolved_at, "resolution_note": a.resolution_note,
            "metadata": json.loads(a.metadata_json) if a.metadata_json else None}


@router.get("/alerts")
def alerts(status: Optional[Literal["OPEN", "ACKNOWLEDGED", "RESOLVED", "UNRESOLVED"]] = "UNRESOLVED",
           severity: Optional[Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]] = None,
           page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100),
           db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.SECURITY_READ))):
    q = db.query(SecurityAlert).filter(SecurityAlert.organization_id == actor.org_id)
    if status == "UNRESOLVED":
        q = q.filter(SecurityAlert.status != "RESOLVED")
    elif status:
        q = q.filter(SecurityAlert.status == status)
    if severity:
        q = q.filter(SecurityAlert.severity == severity)
    total = q.count()
    rows = q.order_by(SecurityAlert.last_seen_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    emails = _emails(db, {x for a in rows for x in (a.subject_id, a.acknowledged_by_user_id, a.resolved_by_user_id)})
    counts = dict(db.query(SecurityAlert.status, func.count()).filter(SecurityAlert.organization_id == actor.org_id)
                  .group_by(SecurityAlert.status).all())
    return {"total": total, "page": page, "page_size": page_size, "counts": counts,
            "alerts": [alert_view(a, emails) for a in rows]}


def _transition(db: Session, actor: Actor, alert_id: str, target: str, note: str, ip: str) -> dict:
    a = db.query(SecurityAlert).filter(SecurityAlert.id == alert_id, SecurityAlert.organization_id == actor.org_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Alert not found")
    if a.status == "RESOLVED":
        raise HTTPException(status_code=409, detail="Alert is already resolved")
    now = datetime.utcnow()
    if target == "ACKNOWLEDGED":
        if a.status == "ACKNOWLEDGED":
            raise HTTPException(status_code=409, detail="Alert is already acknowledged")
        a.acknowledged_by_user_id, a.acknowledged_at = actor.id, now
    else:
        if not note.strip():
            raise HTTPException(status_code=422, detail="A resolution note is required")
        a.resolved_by_user_id, a.resolved_at, a.resolution_note = actor.id, now, note.strip()
        if not a.acknowledged_at:
            a.acknowledged_by_user_id, a.acknowledged_at = actor.id, now
    a.status = target
    db.commit()
    audit(db, actor.id, f"ALERT_{target}", "security_alert", a.id, {"rule": a.rule, "note": note[:300]}, ip,
          org_id=actor.org_id)
    bus.publish("alerts", "alert_updated", {"id": a.id, "status": a.status}, org_id=actor.org_id)
    return alert_view(a, _emails(db, {a.subject_id, a.acknowledged_by_user_id, a.resolved_by_user_id}))


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge(alert_id: str, body: AlertAction, request: Request, db: Session = Depends(get_db),
                actor: Actor = Depends(require(Permission.ALERTS_MANAGE))):
    return _transition(db, actor, alert_id, "ACKNOWLEDGED", body.note, client_ip(request))


@router.post("/alerts/{alert_id}/resolve")
def resolve(alert_id: str, body: AlertAction, request: Request, db: Session = Depends(get_db),
            actor: Actor = Depends(require(Permission.ALERTS_MANAGE))):
    return _transition(db, actor, alert_id, "RESOLVED", body.note, client_ip(request))


@router.get("/audit")
def audit_log(action: Optional[str] = Query(None, max_length=64), page: int = Query(1, ge=1),
              page_size: int = Query(50, ge=1, le=200), db: Session = Depends(get_db),
              actor: Actor = Depends(require(Permission.SECURITY_READ))):
    q = db.query(AuditLog).filter(AuditLog.organization_id == actor.org_id)
    if action:
        q = q.filter(AuditLog.action == action)
    total = q.count()
    rows = q.order_by(AuditLog.timestamp.desc()).offset((page - 1) * page_size).limit(page_size).all()
    emails = _emails(db, {r.actor_user_id for r in rows})
    actions = [a for (a,) in db.query(AuditLog.action).filter(AuditLog.organization_id == actor.org_id)
               .distinct().order_by(AuditLog.action).all()]
    return {"total": total, "page": page, "page_size": page_size, "actions": actions, "entries": [{
        "id": r.id, "timestamp": r.timestamp, "action": r.action, "actor": emails.get(r.actor_user_id),
        "target_type": r.target_type, "target_id": r.target_id, "ip_address": r.ip_address,
        "details": json.loads(r.details_json) if r.details_json else None} for r in rows]}
