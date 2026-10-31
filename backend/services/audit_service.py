"""Security events (what happened) and audit log (who changed what).

Neither ever contains passwords, tokens, images or embeddings.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.core.events import bus
from backend.models.settings import AuditLog
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
    commit: bool = True,
    org_id: Optional[str] = None,
) -> SecurityEvent:
    event = SecurityEvent(
        organization_id=org_id, user_id=user_id, event_type=event_type, description=description[:1000], severity=severity,
        timestamp=datetime.utcnow(), ip_address=ip_address,
        metadata_json=json.dumps(metadata, default=str) if metadata else None,
    )
    db.add(event)
    if commit:
        db.commit()
    log.info("[SECURITY] %s %s (user=%s)", severity, event_type, user_id)
    bus.publish("security", "security_event", {
        "event_type": event_type, "severity": severity, "description": description[:200],
    }, org_id=org_id)
    return event


def audit(
    db: Session,
    actor_id: Optional[str],
    action: str,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    commit: bool = True,
    org_id: Optional[str] = None,
) -> AuditLog:
    row = AuditLog(
        organization_id=org_id, actor_user_id=actor_id, action=action, target_type=target_type, target_id=target_id,
        details_json=json.dumps(details, default=str) if details else None, ip_address=ip_address,
        timestamp=datetime.utcnow(),
    )
    db.add(row)
    if commit:
        db.commit()
    return row
