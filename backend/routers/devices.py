"""Devices, cameras and offline kiosks.

Administration (signed-in members): /api/v1/devices…, /api/v1/cameras…
Device endpoints (signed requests, see device_service): /api/v1/device-api/…
"""
from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.deps import Actor, client_ip, require, require_step_up
from backend.core.errors import ApiError
from backend.core.permissions import Permission
from backend.database import get_db
from backend.models.devices import Camera, Device, DeviceKind, OfflineEvent
from backend.models.settings import AuditLog
from backend.models.user import User
from backend.models.verification import VerificationSession
from backend.services import device_service as svc
from backend.services.audit_service import audit

router = APIRouter(prefix="/api/v1", tags=["devices"])
device_api = APIRouter(prefix="/api/v1/device-api", tags=["devices"])


class DeviceCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    kind: str = Field(DeviceKind.KIOSK, max_length=20)
    location: str = Field("", max_length=200)
    offline_enabled: bool = False
    offline_validity_hours: int = Field(24, ge=1, le=168)


class RevokeBody(BaseModel):
    reason: str = Field("", max_length=200)


class SubjectsBody(BaseModel):
    user_ids: list[str] = Field(default_factory=list, max_length=5000)


class CameraCreate(BaseModel):
    label: str = Field(..., min_length=1, max_length=120)
    location: str = Field("", max_length=200)
    hardware_label: str = Field("", max_length=200)
    capabilities: Optional[dict[str, Any]] = None
    device_id: Optional[str] = Field(None, max_length=36)


class CameraUpdate(BaseModel):
    label: Optional[str] = Field(None, min_length=1, max_length=120)
    location: Optional[str] = Field(None, max_length=200)
    device_id: Optional[str] = Field(None, max_length=36)
    clear_device: bool = False


class CameraHeartbeat(BaseModel):
    hardware_label: Optional[str] = Field(None, max_length=200)
    capabilities: Optional[dict[str, Any]] = None
    state: str = Field("streaming", max_length=20)


class ProvisionBody(BaseModel):
    code: str = Field(..., min_length=6, max_length=32)
    public_key_pem: str = Field(..., max_length=2000)
    software_version: str = Field("", max_length=40)
    platform: str = Field("", max_length=80)
    key_protection: str = Field("software", max_length=32)


def _activity(db: Session, org_id: str, target_type: str, target_id: str, limit: int = 30) -> list[dict]:
    rows = (db.query(AuditLog).filter(AuditLog.organization_id == org_id, AuditLog.target_type == target_type,
                                      AuditLog.target_id == target_id)
            .order_by(AuditLog.timestamp.desc()).limit(limit).all())
    ids = {r.actor_user_id for r in rows if r.actor_user_id}
    names = dict(db.query(User.id, User.full_name).filter(User.id.in_(ids)).all()) if ids else {}
    return [{"at": r.timestamp, "action": r.action, "actor": names.get(r.actor_user_id, "Device / system"),
             "details": json.loads(r.details_json) if r.details_json else None} for r in rows]


# ── devices ─────────────────────────────────────────────────────────────────

@router.get("/devices")
def list_devices(db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.DEVICES_READ))):
    rows = db.query(Device).filter(Device.organization_id == actor.org_id).order_by(Device.created_at.desc()).all()
    return {"devices": [svc.device_view(db, d) for d in rows]}


@router.post("/devices", status_code=201)
def create_device(body: DeviceCreate, request: Request, db: Session = Depends(get_db),
                  actor: Actor = Depends(require(Permission.DEVICES_MANAGE)),
                  _: Actor = Depends(require_step_up(high_risk=True))):
    d, code = svc.create_device(db, actor, body.name, body.kind, body.location, body.offline_enabled,
                                body.offline_validity_hours, client_ip(request))
    return {**svc.device_view(db, d, detail=True), "provisioning_code": code,
            "provisioning_notice": "Enter this one-time code on the device within 15 minutes. It is shown once."}


@router.get("/devices/{device_id}")
def get_device(device_id: str, db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.DEVICES_READ))):
    d = svc.load_device(db, actor, device_id)
    out = svc.device_view(db, d, detail=True)
    out["activity"] = _activity(db, actor.org_id, "device", d.id)
    return out


@router.post("/devices/{device_id}/provisioning-code")
def reissue_code(device_id: str, request: Request, db: Session = Depends(get_db),
                 actor: Actor = Depends(require(Permission.DEVICES_MANAGE)),
                 _: Actor = Depends(require_step_up(high_risk=True))):
    d, code = svc.new_provisioning_code(db, actor, device_id, client_ip(request))
    return {**svc.device_view(db, d), "provisioning_code": code}


@router.post("/devices/{device_id}/revoke")
def revoke_device(device_id: str, body: RevokeBody, request: Request, db: Session = Depends(get_db),
                  actor: Actor = Depends(require(Permission.DEVICES_MANAGE))):
    return svc.device_view(db, svc.revoke(db, actor, device_id, body.reason, client_ip(request)), detail=True)


@router.put("/devices/{device_id}/offline-subjects")
def set_subjects(device_id: str, body: SubjectsBody, request: Request, db: Session = Depends(get_db),
                 actor: Actor = Depends(require(Permission.DEVICES_MANAGE)),
                 _: Actor = Depends(require_step_up(high_risk=True))):
    return svc.device_view(db, svc.set_offline_subjects(db, actor, device_id, body.user_ids, client_ip(request)),
                           detail=True)


@router.get("/devices/{device_id}/offline-events")
def offline_events(device_id: str, db: Session = Depends(get_db),
                   actor: Actor = Depends(require(Permission.DEVICES_READ))):
    d = svc.load_device(db, actor, device_id)
    rows = (db.query(OfflineEvent).filter(OfflineEvent.device_id == d.id)
            .order_by(OfflineEvent.seq.desc()).limit(200).all())
    ids = {r.subject_user_id for r in rows if r.subject_user_id}
    names = dict(db.query(User.id, User.full_name).filter(User.id.in_(ids)).all()) if ids else {}
    return {"events": [{"event_id": r.event_id, "seq": r.seq, "type": r.event_type, "occurred_at": r.occurred_at,
                        "received_at": r.received_at, "subject": names.get(r.subject_user_id),
                        "outcome": r.outcome, "reason_code": r.reason_code, "integrity": r.integrity,
                        "bundle_id": r.bundle_id} for r in rows]}


# ── cameras ─────────────────────────────────────────────────────────────────

@router.get("/cameras")
def list_cameras(include_disabled: bool = True, db: Session = Depends(get_db),
                 actor: Actor = Depends(require(Permission.VERIFY_RUN))):
    q = db.query(Camera).filter(Camera.organization_id == actor.org_id)
    if not include_disabled:
        q = q.filter(Camera.status == "ACTIVE")
    return {"cameras": [svc.camera_view(c) for c in q.order_by(Camera.label).all()],
            "can_manage": actor.can(Permission.CAMERAS_MANAGE)}


@router.post("/cameras", status_code=201)
def create_camera(body: CameraCreate, request: Request, db: Session = Depends(get_db),
                  actor: Actor = Depends(require(Permission.CAMERAS_MANAGE))):
    svc._check_device(db, actor, body.device_id)
    c = Camera(organization_id=actor.org_id, label=body.label.strip(), location=body.location.strip() or None,
               hardware_label=body.hardware_label.strip() or None, device_id=body.device_id,
               capabilities_json=json.dumps(body.capabilities)[:2000] if body.capabilities else None,
               created_by_user_id=actor.id)
    db.add(c)
    db.commit()
    audit(db, actor.id, "CAMERA_REGISTERED", "camera", c.id, {"label": c.label}, client_ip(request), org_id=actor.org_id)
    return svc.camera_view(c)


@router.get("/cameras/{camera_id}")
def get_camera(camera_id: str, db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.CAMERAS_READ))):
    c = svc.load_camera(db, actor, camera_id)
    out = svc.camera_view(c)
    out["activity"] = _activity(db, actor.org_id, "camera", c.id)
    sessions = (db.query(VerificationSession).filter(VerificationSession.camera_id == c.id)
                .order_by(VerificationSession.created_at.desc()).limit(20).all())
    out["recent_sessions"] = [{"session_id": s.id, "created_at": s.created_at, "outcome": s.final_outcome,
                               "purpose": s.purpose, "channel": s.channel} for s in sessions]
    return out


@router.patch("/cameras/{camera_id}")
def update_camera(camera_id: str, body: CameraUpdate, request: Request, db: Session = Depends(get_db),
                  actor: Actor = Depends(require(Permission.CAMERAS_MANAGE))):
    c = svc.load_camera(db, actor, camera_id)
    changes = {}
    if body.label is not None and body.label.strip() != c.label:
        changes["label"] = [c.label, body.label.strip()]
        c.label = body.label.strip()
    if body.location is not None and (body.location.strip() or None) != c.location:
        changes["location"] = [c.location, body.location.strip() or None]
        c.location = body.location.strip() or None
    if body.clear_device or (body.device_id and body.device_id != c.device_id):
        svc._check_device(db, actor, body.device_id)
        changes["device_id"] = [c.device_id, None if body.clear_device else body.device_id]
        c.device_id = None if body.clear_device else body.device_id
    db.commit()
    if changes:
        audit(db, actor.id, "CAMERA_UPDATED", "camera", c.id, changes, client_ip(request), org_id=actor.org_id)
    return svc.camera_view(c)


@router.post("/cameras/{camera_id}/disable")
def disable_camera(camera_id: str, request: Request, db: Session = Depends(get_db),
                   actor: Actor = Depends(require(Permission.CAMERAS_MANAGE))):
    from datetime import datetime
    c = svc.load_camera(db, actor, camera_id)
    c.status, c.disabled_at = "DISABLED", datetime.utcnow()
    db.commit()
    audit(db, actor.id, "CAMERA_DISABLED", "camera", c.id, None, client_ip(request), org_id=actor.org_id)
    return svc.camera_view(c)


@router.post("/cameras/{camera_id}/enable")
def enable_camera(camera_id: str, request: Request, db: Session = Depends(get_db),
                  actor: Actor = Depends(require(Permission.CAMERAS_MANAGE))):
    c = svc.load_camera(db, actor, camera_id)
    c.status, c.disabled_at = "ACTIVE", None
    db.commit()
    audit(db, actor.id, "CAMERA_ENABLED", "camera", c.id, None, client_ip(request), org_id=actor.org_id)
    return svc.camera_view(c)


@router.post("/cameras/{camera_id}/heartbeat")
def camera_heartbeat(camera_id: str, body: CameraHeartbeat, db: Session = Depends(get_db),
                     actor: Actor = Depends(require(Permission.VERIFY_RUN))):
    """Sent by an operator's browser while it streams from this registered camera."""
    from datetime import datetime
    c = svc.load_camera(db, actor, camera_id)
    if c.status != "ACTIVE":
        raise ApiError(409, "CAMERA_DISABLED", "This camera has been disabled by an administrator")
    c.last_heartbeat_at, c.last_heartbeat_by = datetime.utcnow(), actor.email[:64]
    if body.hardware_label and not c.hardware_label:
        c.hardware_label = body.hardware_label[:200]
    if body.capabilities:
        c.capabilities_json = json.dumps(body.capabilities)[:2000]
    db.commit()
    return svc.camera_view(c)


# ── device API (signed) ─────────────────────────────────────────────────────

@device_api.post("/provision", status_code=201)
def provision(body: ProvisionBody, request: Request, db: Session = Depends(get_db)):
    d = svc.provision(db, body.code, body.public_key_pem, body.software_version, body.platform, body.key_protection,
                      client_ip(request))
    from backend.models.organization import Organization
    org = db.query(Organization).filter(Organization.id == d.organization_id).first()
    return {"device_id": d.id, "name": d.name, "kind": d.kind, "organization": org.name if org else None,
            "offline_enabled": d.offline_enabled, "offline_validity_hours": d.offline_validity_hours}


async def signed_device(request: Request, db: Session = Depends(get_db)) -> tuple[Device, dict]:
    body = await request.body()
    if len(body) > 2_000_000:
        raise ApiError(413, "TOO_LARGE", "Request too large")
    h = request.headers
    d = svc.authenticate_request(db, h.get("X-Device-Id", ""), h.get("X-Device-Timestamp", ""),
                                 h.get("X-Device-Nonce", ""), h.get("X-Device-Signature", ""), request.method,
                                 request.url.path, body, client_ip(request))
    try:
        payload = json.loads(body) if body else {}
    except ValueError:
        raise ApiError(422, "INVALID_JSON", "Body must be JSON") from None
    return d, payload


@device_api.post("/heartbeat")
def device_heartbeat(ctx=Depends(signed_device), db: Session = Depends(get_db)):
    d, payload = ctx
    return svc.heartbeat(db, d, payload)


@device_api.post("/rotate-key")
def device_rotate_key(request: Request, ctx=Depends(signed_device), db: Session = Depends(get_db)):
    d, payload = ctx
    d = svc.rotate_key(db, d, str(payload.get("public_key_pem", "")), client_ip(request))
    return {"key_fingerprint": d.key_fingerprint}


@device_api.post("/offline-bundle")
def device_bundle(ctx=Depends(signed_device), db: Session = Depends(get_db)):
    d, _ = ctx
    return svc.build_bundle(db, d)


@device_api.post("/offline-events")
def device_events(request: Request, ctx=Depends(signed_device), db: Session = Depends(get_db)):
    d, payload = ctx
    events = payload.get("events") or []
    if not isinstance(events, list):
        raise ApiError(422, "INVALID_EVENTS", "events must be a list")
    return svc.sync_events(db, d, events, client_ip(request))


@device_api.get("/whoami")
def whoami(ctx=Depends(signed_device)):
    d, _ = ctx
    return {"device_id": d.id, "name": d.name, "status": d.status, "offline_enabled": d.offline_enabled}
