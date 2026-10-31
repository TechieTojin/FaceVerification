"""Dashboard summary and analytics."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.core.deps import Actor, require
from backend.core.permissions import Permission
from backend.database import get_db
from backend.models.user import User
from backend.services import analytics_service, health_service

router = APIRouter(prefix="/api/v1", tags=["dashboard"])


@router.get("/dashboard/summary")
def summary(db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.DASHBOARD_VIEW))):
    data = analytics_service.dashboard_summary(db, actor.org_id)
    ready = health_service.readiness()
    data["health"] = {
        "ready": ready["ready"],
        "database": ready["checks"]["database"],
        "models": ready["checks"]["models"],
        "migrations_up_to_date": ready["checks"]["migrations"]["ok"],
    }
    recent = analytics_service.recent_sessions(db, actor.org_id, 10)
    ids = {x for s in recent for x in (s.target_user_id, s.created_by_user_id) if x}
    names = {u.id: u.full_name for u in db.query(User).filter(User.id.in_(ids)).all()} if ids else {}
    show_names = actor.can(Permission.HISTORY_READ_ALL) or actor.can(Permission.HISTORY_READ_OWN)
    data["recent_sessions"] = [{
        "session_id": s.id, "purpose": s.purpose, "outcome": s.final_outcome, "reason_code": s.reason_code,
        "completed_at": s.completed_at, "processing_ms": round(s.total_processing_ms or 0, 1),
        "target_name": names.get(s.target_user_id) if show_names else None,
        "operator_name": names.get(s.created_by_user_id) if show_names else None,
    } for s in recent]
    return data


@router.get("/analytics/overview")
def analytics_overview(days: int = Query(30, ge=1, le=365), db: Session = Depends(get_db),
                       actor: Actor = Depends(require(Permission.ANALYTICS_READ))):
    data = analytics_service.overview(db, actor.org_id, days)
    from backend.services.inference import runtime
    data["models"] = runtime.snapshot()["models"]
    data["runtime_latency_ms"] = runtime.snapshot()["latency_ms"]
    return data
