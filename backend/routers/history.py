"""Verification history: search, detail, CSV export (role-scoped)."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import or_
from sqlalchemy.orm import Query as SAQuery, Session

from backend.core.deps import Actor, client_ip, get_actor
from backend.core.permissions import Permission
from backend.database import get_db
from backend.models.user import User
from backend.models.verification import LivenessChallenge, VerificationEvent, VerificationSession
from backend.services.audit_service import audit

router = APIRouter(prefix="/api/v1/history", tags=["history"])

SORTS = {
    "newest": VerificationSession.created_at.desc(),
    "oldest": VerificationSession.created_at.asc(),
    "duration": VerificationSession.total_processing_ms.desc(),
    "distance": VerificationSession.final_distance.asc(),
}


def _scoped(db: Session, actor: Actor) -> SAQuery:
    base = db.query(VerificationSession).filter(VerificationSession.organization_id == actor.org_id)
    if actor.can(Permission.HISTORY_READ_ALL):
        return base
    if actor.can(Permission.HISTORY_READ_OWN):
        return base.filter(VerificationSession.created_by_user_id == actor.id)
    raise HTTPException(status_code=403, detail="You do not have permission to view history")


def _filtered(db: Session, actor: Actor, purpose, outcome, target_user_id, operator_id, search, date_from, date_to):
    q = _scoped(db, actor)
    if purpose:
        q = q.filter(VerificationSession.purpose == purpose)
    if outcome:
        q = q.filter(VerificationSession.final_outcome == outcome)
    if target_user_id:
        q = q.filter(VerificationSession.target_user_id == target_user_id)
    if operator_id:
        q = q.filter(VerificationSession.created_by_user_id == operator_id)
    if date_from:
        q = q.filter(VerificationSession.created_at >= date_from)
    if date_to:
        q = q.filter(VerificationSession.created_at < date_to + timedelta(days=1))
    if search:
        term = f"%{search.strip()[:100].replace('%', '').replace('_', '')}%"
        ids = [u.id for u in db.query(User.id).filter(or_(User.email.ilike(term), User.full_name.ilike(term))).all()]
        q = q.filter(or_(VerificationSession.id.ilike(term), VerificationSession.target_user_id.in_(ids),
                         VerificationSession.created_by_user_id.in_(ids)))
    return q


def row_view(s: VerificationSession, names: dict[str, tuple[str, str]], scores: bool) -> dict:
    tgt = names.get(s.target_user_id or "", (None, None))
    op = names.get(s.created_by_user_id or "", (None, None))
    v = {
        "session_id": s.id, "purpose": s.purpose, "created_at": s.created_at, "completed_at": s.completed_at,
        "channel": s.channel, "camera_id": s.camera_id, "review_status": s.review_status,
        "status": "ACTIVE" if s.is_active else "CLOSED", "outcome": s.final_outcome, "reason_code": s.reason_code,
        "target_user_id": s.target_user_id, "target_name": tgt[0], "target_email": tgt[1],
        "operator_user_id": s.created_by_user_id, "operator_name": op[0], "operator_email": op[1],
        "frames": s.frame_count, "attempts_used": s.attempts_used, "rounds": s.challenge_round + 1,
        "processing_ms": round(s.total_processing_ms or 0.0, 1), "policy_version": s.policy_version,
        "duration_s": round((s.completed_at - s.created_at).total_seconds(), 1) if s.completed_at else None,
        "liveness_outcome": ("FAILED" if s.final_outcome == "LIVENESS_FAILED"
                             else "PASSED" if s.final_outcome in ("VERIFIED", "NOT_MATCHED", "ENROLMENT_CAPTURED") else None),
    }
    if scores:
        v["final_distance"] = s.final_distance
        v["threshold"] = s.threshold
    return v


def _names(db: Session, sessions: list[VerificationSession]) -> dict[str, tuple[str, str]]:
    ids = {x for s in sessions for x in (s.target_user_id, s.created_by_user_id) if x}
    if not ids:
        return {}
    return {u.id: (u.full_name, u.email) for u in db.query(User).filter(User.id.in_(ids)).all()}


Purpose = Optional[Literal["VERIFY", "ENROL", "DOCUMENT"]]


@router.get("/sessions")
def list_sessions(purpose: Purpose = None, outcome: Optional[str] = Query(None, max_length=32),
                  target_user_id: Optional[str] = None, operator_id: Optional[str] = None,
                  search: Optional[str] = Query(None, max_length=100),
                  date_from: Optional[datetime] = None, date_to: Optional[datetime] = None,
                  sort: Literal["newest", "oldest", "duration", "distance"] = "newest",
                  page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
                  db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    scores = actor.can(Permission.HISTORY_SCORES)
    if sort == "distance" and not scores:
        sort = "newest"
    q = _filtered(db, actor, purpose, outcome, target_user_id, operator_id, search, date_from, date_to)
    total = q.count()
    rows = q.order_by(SORTS[sort]).offset((page - 1) * page_size).limit(page_size).all()
    names = _names(db, rows)
    return {"total": total, "page": page, "page_size": page_size,
            "sessions": [row_view(s, names, scores) for s in rows]}


@router.get("/sessions/{session_id}")
def session_detail(session_id: str, db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    s = _scoped(db, actor).filter(VerificationSession.id == session_id).first()
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    scores = actor.can(Permission.HISTORY_SCORES)
    names = _names(db, [s])
    events = (db.query(VerificationEvent).filter(VerificationEvent.session_id == s.id)
              .order_by(VerificationEvent.timestamp.asc(), VerificationEvent.frame_seq.asc()).all())
    challenges = (db.query(LivenessChallenge).filter(LivenessChallenge.session_id == s.id)
                  .order_by(LivenessChallenge.round, LivenessChallenge.step_index).all())
    out = row_view(s, names, scores)
    out["expires_at"] = s.expires_at
    out["challenges"] = [{"round": c.round, "index": c.step_index, "type": c.step_type, "status": c.status,
                          "started_at": c.started_at, "completed_at": c.completed_at,
                          "evidence": json.loads(c.evidence_json) if (c.evidence_json and scores) else None}
                         for c in challenges]
    out["events"] = []
    for e in events:
        ev = {"timestamp": e.timestamp, "seq": e.frame_seq, "outcome": e.outcome, "reason_code": e.reason_code,
              "step_type": e.step_type, "face_count": e.face_count, "quality_score": e.quality_score,
              "processing_ms": e.inference_ms, "explanation": e.explanation, "policy_version": e.policy_version}
        if scores:
            ev.update({"distance": e.distance, "threshold": e.threshold, "passive_label": e.passive_label,
                       "passive_confidence": e.passive_confidence, "yaw": e.yaw, "pitch": e.pitch})
        out["events"].append(ev)
    if s.policy_snapshot and actor.can(Permission.POLICY_READ):
        out["policy_snapshot"] = json.loads(s.policy_snapshot)["values"]
    return out


@router.get("/export.csv")
def export_csv(request: Request, purpose: Purpose = None, outcome: Optional[str] = Query(None, max_length=32),
               target_user_id: Optional[str] = None, operator_id: Optional[str] = None,
               search: Optional[str] = Query(None, max_length=100),
               date_from: Optional[datetime] = None, date_to: Optional[datetime] = None,
               db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    if not (actor.can(Permission.HISTORY_EXPORT) or actor.can(Permission.HISTORY_READ_ALL)):
        raise HTTPException(status_code=403, detail="You do not have permission to export history")
    scores = actor.can(Permission.HISTORY_SCORES)
    q = _filtered(db, actor, purpose, outcome, target_user_id, operator_id, search, date_from, date_to)
    rows = q.order_by(VerificationSession.created_at.desc()).limit(10_000).all()
    names = _names(db, rows)
    cols = ["session_id", "purpose", "created_at", "completed_at", "outcome", "reason_code", "target_name",
            "target_email", "operator_email", "frames", "attempts_used", "duration_s", "processing_ms",
            "liveness_outcome", "policy_version"] + (["final_distance", "threshold"] if scores else [])
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for s in rows:
        r = row_view(s, names, scores)
        # neutralise spreadsheet formula injection in user-controlled text
        for k in ("target_name", "target_email", "operator_email"):
            if isinstance(r.get(k), str) and r[k][:1] in ("=", "+", "-", "@"):
                r[k] = "'" + r[k]
        w.writerow(r)
    audit(db, actor.id, "HISTORY_EXPORTED", "history", None, {"rows": len(rows)}, client_ip(request),
          org_id=actor.org_id)
    fname = f"verification-history-{datetime.utcnow():%Y%m%d-%H%M%S}.csv"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{fname}"'})
