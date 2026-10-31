"""Verification policy (admin settings) — versioned, validated, audited, enforced."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.deps import Actor, client_ip, require
from backend.core.events import bus
from backend.core.permissions import Permission
from backend.database import get_db
from backend.models.settings import PolicyVersion
from backend.models.user import User
from backend.services import policy_service
from backend.services.audit_service import audit, log_security_event

router = APIRouter(prefix="/api/v1/policy", tags=["policy"])


class PolicyUpdate(BaseModel):
    changes: dict[str, Any] = Field(..., min_length=1)
    confirm: bool = False
    reason: str = Field("", max_length=500)


def _version_view(row: PolicyVersion, emails: dict) -> dict:
    return {"version": row.version, "created_at": row.created_at, "created_by": emails.get(row.created_by_user_id),
            "change_summary": row.change_summary, "is_sensitive_change": row.is_sensitive_change}


@router.get("")
def get_policy(db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.POLICY_READ))):
    p = policy_service.get_active_policy(db, actor.org_id)
    row = (db.query(PolicyVersion)
           .filter(PolicyVersion.organization_id == actor.org_id, PolicyVersion.version == p.version).first())
    return {"version": p.version, "updated_at": row.created_at if row else None, "values": p.as_dict(),
            "specs": policy_service.describe_specs(),
            "active_sessions_note": "Sessions already in progress keep the policy version they started with."}


@router.get("/versions")
def versions(db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.POLICY_READ))):
    rows = (db.query(PolicyVersion).filter(PolicyVersion.organization_id == actor.org_id)
            .order_by(PolicyVersion.version.desc()).limit(100).all())
    ids = {r.created_by_user_id for r in rows if r.created_by_user_id}
    emails = {u.id: u.email for u in db.query(User).filter(User.id.in_(ids)).all()} if ids else {}
    return {"versions": [_version_view(r, emails) for r in rows]}


@router.get("/versions/{version}")
def version_detail(version: int, db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.POLICY_READ))):
    row = (db.query(PolicyVersion)
           .filter(PolicyVersion.organization_id == actor.org_id, PolicyVersion.version == version).first())
    if not row:
        raise HTTPException(status_code=404, detail="Policy version not found")
    return {**_version_view(row, {}), "values": json.loads(row.values_json)}


@router.put("")
def update_policy(body: PolicyUpdate, request: Request, db: Session = Depends(get_db),
                  actor: Actor = Depends(require(Permission.POLICY_MANAGE))):
    ip = client_ip(request)
    try:
        policy, diff, weakening = policy_service.update_policy(db, actor.org_id, body.changes, actor.id, body.confirm,
                                                               body.reason)
    except policy_service.PolicyValidationError as exc:
        raise HTTPException(status_code=422, detail={"message": "Invalid settings", "errors": exc.errors}) from exc
    except policy_service.SensitiveChangeNotConfirmed as exc:
        raise HTTPException(status_code=409, detail={
            "message": "These changes weaken security and must be confirmed", "code": "CONFIRMATION_REQUIRED",
            "sensitive_changes": exc.changes}) from exc
    if not diff:
        return {"version": policy.version, "changed": {}, "message": "No changes"}
    changed = {k: {"from": o, "to": n} for k, (o, n) in diff.items()}
    audit(db, actor.id, "POLICY_UPDATED", "policy", str(policy.version),
          {"changes": changed, "reason": body.reason, "sensitive": weakening}, ip, org_id=actor.org_id)
    log_security_event(db, "POLICY_UPDATED",
                       f"Policy v{policy.version}: {', '.join(diff)}" + (" (weakening, confirmed)" if weakening else ""),
                       "HIGH" if weakening else "WARNING", user_id=actor.id, ip_address=ip,
                       metadata={"version": policy.version, "sensitive": weakening}, org_id=actor.org_id)
    bus.publish("policy", "policy_updated", {"version": policy.version}, org_id=actor.org_id)
    return {"version": policy.version, "changed": changed, "sensitive": weakening,
            "message": f"Policy updated to version {policy.version}"}
