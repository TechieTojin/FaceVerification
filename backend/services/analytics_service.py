"""Operational analytics computed from stored sessions/events (no fabricated numbers)."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models.verification import SecurityAlert, VerificationEvent, VerificationSession

DECIDED = ("VERIFIED", "NOT_MATCHED", "LIVENESS_FAILED")


def percentiles(values: Iterable[float]) -> dict[str, Optional[float]]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"p50": None, "p95": None, "p99": None, "avg": None, "samples": 0}
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return {"p50": round(float(p50), 1), "p95": round(float(p95), 1), "p99": round(float(p99), 1),
            "avg": round(float(arr.mean()), 1), "samples": int(arr.size)}


def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def dashboard_summary(db: Session, org_id: str) -> dict[str, Any]:
    now = datetime.utcnow()
    day = now - timedelta(hours=24)
    hour = now - timedelta(hours=1)
    active = (db.query(VerificationSession.purpose, func.count())
              .filter(VerificationSession.organization_id == org_id, VerificationSession.is_active == True)  # noqa: E712
              .group_by(VerificationSession.purpose).all())
    outcomes = dict(db.query(VerificationSession.final_outcome, func.count())
                    .filter(VerificationSession.organization_id == org_id, VerificationSession.purpose == "VERIFY",
                            VerificationSession.completed_at >= day)
                    .group_by(VerificationSession.final_outcome).all())
    completed = sum(v for k, v in outcomes.items() if k)
    lat = [r[0] for r in db.query(VerificationEvent.inference_ms)
           .filter(VerificationEvent.organization_id == org_id, VerificationEvent.timestamp >= hour,
                   VerificationEvent.inference_ms.isnot(None)).all()]
    sessions_last_hour = (db.query(VerificationSession)
                          .filter(VerificationSession.organization_id == org_id,
                                  VerificationSession.created_at >= hour).count())
    frames_last_hour = len(lat)
    open_alerts = dict(db.query(SecurityAlert.severity, func.count())
                       .filter(SecurityAlert.organization_id == org_id, SecurityAlert.status != "RESOLVED")
                       .group_by(SecurityAlert.severity).all())
    suspicious = (db.query(SecurityAlert)
                  .filter(SecurityAlert.organization_id == org_id, SecurityAlert.last_seen_at >= day).count())
    from backend.models.reviews import OPEN_REVIEW_STATUSES, VerificationReview
    pending_reviews = (db.query(VerificationReview)
                       .filter(VerificationReview.organization_id == org_id,
                               VerificationReview.status.in_(OPEN_REVIEW_STATUSES)).count())
    return {
        "generated_at": now.isoformat() + "Z",
        "active_sessions": {k: v for k, v in active},
        "active_sessions_total": sum(v for _, v in active),
        "window_hours": 24,
        "completed_verifications": completed,
        "verified": outcomes.get("VERIFIED", 0),
        "not_matched": outcomes.get("NOT_MATCHED", 0),
        "liveness_failures": outcomes.get("LIVENESS_FAILED", 0),
        "other_outcomes": {k: v for k, v in outcomes.items() if k and k not in DECIDED},
        "open_alerts": open_alerts,
        "open_alerts_total": sum(open_alerts.values()),
        "suspicious_events_24h": suspicious,
        "pending_reviews": pending_reviews,
        "latency_last_hour_ms": percentiles(lat),
        "throughput_last_hour": {"sessions": sessions_last_hour, "frames": frames_last_hour,
                                 "frames_per_minute": round(frames_last_hour / 60.0, 2)},
    }


def recent_sessions(db: Session, org_id: str, limit: int = 10) -> list[VerificationSession]:
    return (db.query(VerificationSession)
            .filter(VerificationSession.organization_id == org_id, VerificationSession.is_active == False)  # noqa: E712
            .order_by(VerificationSession.completed_at.desc()).limit(limit).all())


def overview(db: Session, org_id: str, days: int) -> dict[str, Any]:
    now = datetime.utcnow()
    since = now - timedelta(days=days)
    sessions = (db.query(VerificationSession.purpose, VerificationSession.final_outcome, VerificationSession.created_at,
                         VerificationSession.completed_at, VerificationSession.is_active)
                .filter(VerificationSession.organization_id == org_id, VerificationSession.created_at >= since).all())
    verify = [s for s in sessions if s.purpose == "VERIFY"]
    closed = [s for s in verify if not s.is_active]
    decided = [s for s in closed if s.final_outcome in DECIDED]
    outcome_dist = Counter(s.final_outcome for s in closed)

    events = (db.query(VerificationEvent.timestamp, VerificationEvent.outcome, VerificationEvent.inference_ms)
              .filter(VerificationEvent.organization_id == org_id, VerificationEvent.timestamp >= since)
              .limit(200_000).all())
    frame_events = [e for e in events if e.inference_ms is not None]
    frame_outcomes = Counter(e.outcome for e in frame_events)

    by_day: dict[str, dict[str, Any]] = defaultdict(lambda: {"sessions": 0, "verified": 0, "not_matched": 0,
                                                             "liveness_failed": 0, "errors": 0, "latencies": []})
    for s in verify:
        d = by_day[s.created_at.strftime("%Y-%m-%d")]
        d["sessions"] += 1
        if s.final_outcome == "VERIFIED":
            d["verified"] += 1
        elif s.final_outcome == "NOT_MATCHED":
            d["not_matched"] += 1
        elif s.final_outcome == "LIVENESS_FAILED":
            d["liveness_failed"] += 1
    for e in frame_events:
        d = by_day[e.timestamp.strftime("%Y-%m-%d")]
        d["latencies"].append(e.inference_ms)
        if e.outcome == "ERROR":
            d["errors"] += 1
    daily = []
    for i in range(days, -1, -1):
        key = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        d = by_day.get(key, {"sessions": 0, "verified": 0, "not_matched": 0, "liveness_failed": 0, "errors": 0,
                             "latencies": []})
        p = percentiles(d["latencies"])
        daily.append({"date": key, "sessions": d["sessions"], "verified": d["verified"], "not_matched": d["not_matched"],
                      "liveness_failed": d["liveness_failed"], "frame_errors": d["errors"],
                      "latency_p50": p["p50"], "latency_p95": p["p95"], "frames": p["samples"]})

    hourly = Counter(s.created_at.replace(minute=0, second=0, microsecond=0) for s in sessions
                     if s.created_at >= now - timedelta(hours=24))
    throughput = [{"hour": (now - timedelta(hours=h)).replace(minute=0, second=0, microsecond=0).isoformat(),
                   "sessions": hourly.get((now - timedelta(hours=h)).replace(minute=0, second=0, microsecond=0), 0)}
                  for h in range(23, -1, -1)]

    return {
        "days": days,
        "totals": {"sessions": len(verify), "closed": len(closed), "decided": len(decided),
                   "frames": len(frame_events)},
        "session_completion_rate": _rate(len(decided), len(closed)),
        "verified_rate": _rate(outcome_dist.get("VERIFIED", 0), len(decided)),
        "liveness_rejection_rate": _rate(outcome_dist.get("LIVENESS_FAILED", 0), len(decided)),
        "quality_failure_rate": _rate(frame_outcomes.get("QUALITY_FAILED", 0), len(frame_events)),
        "frame_error_rate": _rate(frame_outcomes.get("ERROR", 0), len(frame_events)),
        "outcome_distribution": dict(outcome_dist),
        "frame_outcome_distribution": dict(frame_outcomes),
        "latency_ms": percentiles(e.inference_ms for e in frame_events),
        "daily": daily,
        "throughput_24h": throughput,
        "notes": [
            "Rates are operational outcomes, not accuracy. FAR/FRR require labelled evaluation data "
            "(see Model performance).",
            "Latency is end-to-end server processing per frame, including decoding and all model calls.",
        ],
    }
