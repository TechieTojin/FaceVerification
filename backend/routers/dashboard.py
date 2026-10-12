"""Dashboard summary router."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend.models.user import User
from backend.models.biometric import BiometricEnrolment
from backend.models.verification import VerificationEvent, VerificationSession, SecurityEvent, OutcomeEnum
from backend.routers.auth import get_current_user

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/summary")
def get_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    total_users = db.query(User).count()
    active_enrolments = db.query(BiometricEnrolment).filter(
        BiometricEnrolment.is_active == True,
        BiometricEnrolment.enrolled_at.isnot(None),
    ).count()

    total_events = db.query(VerificationEvent).count()
    successful = db.query(VerificationEvent).filter(VerificationEvent.outcome == OutcomeEnum.VERIFIED).count()
    failed = db.query(VerificationEvent).filter(VerificationEvent.outcome == OutcomeEnum.REJECTED).count()
    spoof_attempts = db.query(VerificationEvent).filter(VerificationEvent.outcome == OutcomeEnum.SPOOF_DETECTED).count()

    success_rate = (successful / total_events * 100) if total_events > 0 else 0.0

    avg_latency = db.query(func.avg(VerificationEvent.inference_ms)).scalar() or 0.0

    # Recent events
    recent = db.query(VerificationEvent).order_by(VerificationEvent.timestamp.desc()).limit(10).all()
    recent_events = [
        {
            "id": e.id,
            "timestamp": e.timestamp.isoformat(),
            "outcome": e.outcome.value,
            "user_id": e.user_id,
            "explanation": e.explanation,
            "inference_ms": e.inference_ms,
        }
        for e in recent
    ]

    return {
        "total_users": total_users,
        "active_enrolments": active_enrolments,
        "total_verifications": total_events,
        "successful_verifications": successful,
        "failed_verifications": failed,
        "spoof_attempts": spoof_attempts,
        "success_rate": round(success_rate, 1),
        "avg_latency_ms": round(float(avg_latency), 1),
        "recent_events": recent_events,
        "backend_healthy": True,
        "models_ready": True,
    }
