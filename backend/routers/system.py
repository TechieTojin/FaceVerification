"""Health, readiness, system status and the live event stream."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backend.config import settings
from backend.core.deps import Actor, get_actor, require
from backend.core.events import bus
from backend.core.permissions import Permission
from backend.models.user import User
from backend.services import health_service

router = APIRouter(prefix="/api/v1", tags=["system"])

HEARTBEAT_S = 15

# channel -> permission required (None = any member; such channels only carry events addressed to the caller)
CHANNEL_PERMISSIONS: dict[str, Permission | None] = {
    "sessions": Permission.DASHBOARD_VIEW,
    "enrolment": Permission.ENROL_READ,
    "alerts": Permission.SECURITY_READ,
    "security": Permission.SECURITY_READ,
    "policy": Permission.POLICY_READ,
    "reviews": Permission.REVIEW_READ,
    "devices": Permission.DEVICES_READ,
    "notifications": None,
    "pairing": None,
}


@router.get("/health")
def health():
    """Liveness: the process is up and serving requests."""
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


@router.get("/health/ready")
def ready():
    """Readiness: database reachable, schema current, models loaded."""
    data = health_service.readiness()
    return JSONResponse(status_code=200 if data["ready"] else 503, content=json.loads(json.dumps(data, default=str)))


@router.get("/system/health")
def system_health(_: Actor = Depends(require(Permission.SYSTEM_HEALTH))):
    return health_service.system_health()


@router.get("/stream")
async def stream(request: Request, channels: str = Query("sessions,alerts", max_length=200),
                 actor: Actor = Depends(get_actor)):
    """Server-Sent Events, scoped to the caller's active organisation and filtered by permission.

    Organisation-wide channels need the listed permission; user-addressed channels
    (notifications, pairing) only ever deliver events addressed to the caller.
    """
    wanted = {c.strip() for c in channels.split(",") if c.strip()}
    allowed = {c for c in wanted if c in CHANNEL_PERMISSIONS
               and (CHANNEL_PERMISSIONS[c] is None or actor.can(CHANNEL_PERMISSIONS[c]))}
    sub = bus.subscribe(allowed, actor.org_id, actor.id)

    async def gen():
        try:
            yield f"event: hello\ndata: {json.dumps({'channels': sorted(allowed)})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(sub.queue.get(), timeout=HEARTBEAT_S)
                    event = json.loads(payload)["event"]
                    yield f"event: {event}\ndata: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield f"event: heartbeat\ndata: {{\"dropped\": {sub.dropped}}}\n\n"
        finally:
            bus.unsubscribe(sub)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})
