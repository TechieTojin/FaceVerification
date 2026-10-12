"""Analytics router."""
from __future__ import annotations
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend.models.verification import VerificationEvent, OutcomeEnum
from backend.routers.auth import get_current_user
from backend.models.user import User

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("")
def get_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Verifications by day (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    daily_counts = (
        db.query(
            func.date(VerificationEvent.timestamp).label("date"),
            func.count().label("count"),
        )
        .filter(VerificationEvent.timestamp >= thirty_days_ago)
        .group_by(func.date(VerificationEvent.timestamp))
        .order_by(func.date(VerificationEvent.timestamp))
        .all()
    )

    verifications_by_day = [
        {"date": str(row.date), "count": row.count}
        for row in daily_counts
    ]

    # Outcome distribution
    outcome_counts = (
        db.query(VerificationEvent.outcome, func.count().label("count"))
        .group_by(VerificationEvent.outcome)
        .all()
    )
    outcome_distribution = {
        row.outcome.value: row.count for row in outcome_counts
    }

    # Failure reasons
    failure_reasons = {}
    for outcome in [OutcomeEnum.REJECTED, OutcomeEnum.NO_FACE, OutcomeEnum.SPOOF_DETECTED, OutcomeEnum.ERROR]:
        count = db.query(VerificationEvent).filter(VerificationEvent.outcome == outcome).count()
        if count > 0:
            failure_reasons[outcome.value] = count

    # Latency histogram buckets
    all_latencies = [e.inference_ms for e in db.query(VerificationEvent).filter(VerificationEvent.inference_ms.isnot(None)).all()]
    buckets = {"<500ms": 0, "500ms-1s": 0, "1s-2s": 0, "2s-5s": 0, ">5s": 0}
    for lat in all_latencies:
        if lat < 500:
            buckets["<500ms"] += 1
        elif lat < 1000:
            buckets["500ms-1s"] += 1
        elif lat < 2000:
            buckets["1s-2s"] += 1
        elif lat < 5000:
            buckets["2s-5s"] += 1
        else:
            buckets[">5s"] += 1

    return {
        "verifications_by_day": verifications_by_day,
        "outcome_distribution": outcome_distribution,
        "failure_reasons": failure_reasons,
        "latency_histogram": buckets,
    }
