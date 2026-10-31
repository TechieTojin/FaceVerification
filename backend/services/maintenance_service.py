"""Periodic maintenance: expire sessions and short-lived records, deliver queued work, enforce retention.

Retention is per organisation (each organisation's policy governs its own
rows). Global tables (replay digests, login attempts) keep the *longest*
period any organisation asks for.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Callable

from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.models.auth import LoginAttempt, RefreshToken
from backend.models.organization import Organization
from backend.models.verification import FrameDigest, VerificationEvent, VerificationSession
from backend.services import capture_service, document_service
from backend.services.policy_service import Policy, get_active_policy

log = logging.getLogger(__name__)

last_run: dict[str, Any] = {"at": None, "ok": None, "result": None, "error": None}

# Feature services register periodic tasks here: fn(db, now, policies_by_org) -> {counter: n}
TASKS: list[Callable[[Session, datetime, dict[str, Policy]], dict[str, int]]] = []


def _org_retention(db: Session, now: datetime, policies: dict[str, Policy]) -> dict[str, int]:
    events = sessions = 0
    for org_id, policy in policies.items():
        events += (db.query(VerificationEvent)
                   .filter(VerificationEvent.organization_id == org_id,
                           VerificationEvent.timestamp < now - timedelta(days=int(policy.retention_events_days)))
                   .delete(synchronize_session=False))
        old = (db.query(VerificationSession)
               .filter(VerificationSession.organization_id == org_id, VerificationSession.is_active == False,  # noqa: E712
                       VerificationSession.completed_at < now - timedelta(days=int(policy.retention_sessions_days)))
               .all())
        for s in old:
            db.delete(s)   # cascades to events and liveness challenges
        sessions += len(old)
    return {"events_deleted": events, "sessions_deleted": sessions}


def run_maintenance() -> dict[str, int]:
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        policies = {o.id: get_active_policy(db, o.id) for o in db.query(Organization).all()}
        digest_h = max((int(p.retention_frame_digest_hours) for p in policies.values()), default=48)
        attempts_d = max((int(p.retention_login_attempts_days) for p in policies.values()), default=30)
        out = {"sessions_expired": capture_service.expire_stale_sessions(db)}
        out["documents_purged"] = document_service.purge_expired(db)
        out["frame_digests_deleted"] = (db.query(FrameDigest)
                                        .filter(FrameDigest.created_at < now - timedelta(hours=digest_h))
                                        .delete(synchronize_session=False))
        out.update(_org_retention(db, now, policies))
        out["login_attempts_deleted"] = (db.query(LoginAttempt)
                                         .filter(LoginAttempt.attempted_at < now - timedelta(days=attempts_d))
                                         .delete(synchronize_session=False))
        out["refresh_tokens_deleted"] = (db.query(RefreshToken)
                                         .filter(RefreshToken.expires_at < now - timedelta(days=1))
                                         .delete(synchronize_session=False))
        db.commit()
        for task in TASKS:
            try:
                out.update(task(db, now, policies))
                db.commit()
            except Exception:  # noqa: BLE001 — one failing task must not stop the others
                db.rollback()
                log.exception("Maintenance task %s failed", getattr(task, "__name__", task))
        last_run.update(at=now, ok=True, result=out, error=None)
        if any(out.values()):
            log.info("Maintenance: %s", out)
        return out
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        log.exception("Maintenance failed")
        last_run.update(at=datetime.utcnow(), ok=False, error=type(exc).__name__)
        return {}
    finally:
        db.close()
