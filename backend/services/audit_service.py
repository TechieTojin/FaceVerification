"""Audit logging service."""
from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from backend.models.verification import SecurityEvent

log = logging.getLogger(__name__)


def log_security_event(
    db: Session,
    event_type: str,
    description: str,
    severity: str = "INFO",
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> SecurityEvent:
    """Write a security event to the database."""
    event = SecurityEvent(
        user_id=user_id,
        event_type=event_type,
        description=description,
        severity=severity,
        timestamp=datetime.utcnow(),
        ip_address=ip_address,
        metadata_json=json.dumps(metadata) if metadata else None,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    log.info("[AUDIT] %s: %s (user=%s)", event_type, description, user_id)
    return event
