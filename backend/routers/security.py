"""Security events router."""
from __future__ import annotations
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
import json

from backend.database import get_db
from backend.models.verification import SecurityEvent
from backend.models.user import User
from backend.routers.auth import get_current_user, require_admin

router = APIRouter(prefix="/api/v1/security", tags=["security"])


@router.get("/events")
def list_security_events(
    skip: int = 0,
    limit: int = 50,
    severity: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    q = db.query(SecurityEvent)
    if severity:
        q = q.filter(SecurityEvent.severity == severity)
    total = q.count()
    events = q.order_by(SecurityEvent.timestamp.desc()).offset(skip).limit(limit).all()

    return {
        "total": total,
        "events": [
            {
                "id": e.id,
                "event_type": e.event_type,
                "description": e.description,
                "severity": e.severity,
                "timestamp": e.timestamp.isoformat(),
                "user_id": e.user_id,
                "ip_address": e.ip_address,
                "metadata": json.loads(e.metadata_json) if e.metadata_json else None,
            }
            for e in events
        ],
    }
