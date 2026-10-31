"""Integration API.

Administration (signed-in administrators, `integrations:manage`):
  /api/v1/integrations/clients…        clients, credentials, webhooks, deliveries
Client API (credential `Authorization: Bearer fvk_…`, scoped, rate-limited):
  POST /api/v1/integrations/sessions            create a hosted verification (Idempotency-Key supported)
  GET  /api/v1/integrations/sessions/{id}       status + signed assertion when finished
  POST /api/v1/integrations/assertions/introspect   one-time server-side validation
Public:
  GET  /api/v1/integrations/.well-known/jwks.json
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.deps import Actor, client_ip, require, require_step_up
from backend.core.errors import ApiError
from backend.core.permissions import Permission
from backend.database import get_db
from backend.models.integrations import IntegrationClient, WebhookDelivery
from backend.services import integration_service as svc

router = APIRouter(prefix="/api/v1/integrations", tags=["integrations"])


class ClientCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    description: str = Field("", max_length=2000)
    scopes: list[str] = Field(default_factory=lambda: ["sessions:write", "sessions:read", "assertions:introspect"])
    rate_limit_per_minute: int = Field(60, ge=1, le=6000)


class RotateBody(BaseModel):
    grace_hours: int = Field(0, ge=0, le=72)


class WebhookCreate(BaseModel):
    url: str = Field(..., max_length=500)
    events: list[str] = Field(default_factory=lambda: ["verification.completed", "verification.reviewed"])


class ApiSessionCreate(BaseModel):
    subject_user_id: Optional[str] = Field(None, max_length=36)
    subject_email: Optional[str] = Field(None, max_length=255)
    external_ref: Optional[str] = Field(None, max_length=128)
    expires_in: int = Field(600, ge=60, le=1800)


class IntrospectBody(BaseModel):
    assertion: str = Field(..., max_length=8000)


def api_principal(request: Request, db: Session = Depends(get_db),
                  authorization: str = Header("", max_length=300)) -> svc.ApiPrincipal:
    raw = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    return svc.authenticate(db, raw, client_ip(request))


# ── public ──────────────────────────────────────────────────────────────────

@router.get("/.well-known/jwks.json")
def jwks(db: Session = Depends(get_db)):
    return svc.jwks(db)


# ── administration ──────────────────────────────────────────────────────────

@router.get("/clients")
def list_clients(db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.INTEGRATIONS_MANAGE))):
    rows = (db.query(IntegrationClient).filter(IntegrationClient.organization_id == actor.org_id)
            .order_by(IntegrationClient.created_at.desc()).all())
    return {"clients": [svc.client_view(db, c) for c in rows], "scopes": list(svc.SCOPES), "events": list(svc.EVENTS),
            "issuer": svc.issuer()}


@router.post("/clients", status_code=201)
def create_client(body: ClientCreate, request: Request, db: Session = Depends(get_db),
                  actor: Actor = Depends(require(Permission.INTEGRATIONS_MANAGE)),
                  _: Actor = Depends(require_step_up(high_risk=True))):
    client, raw = svc.create_client(db, actor, body.name, body.description, body.scopes, body.rate_limit_per_minute,
                                    client_ip(request))
    return {**svc.client_view(db, client, detail=True), "api_key": raw,
            "api_key_notice": "Store this key now. It will not be shown again."}


@router.get("/clients/{client_id}")
def get_client(client_id: str, db: Session = Depends(get_db),
               actor: Actor = Depends(require(Permission.INTEGRATIONS_MANAGE))):
    return svc.client_view(db, svc.load_client(db, actor, client_id), detail=True)


@router.post("/clients/{client_id}/credentials/rotate", status_code=201)
def rotate(client_id: str, body: RotateBody, request: Request, db: Session = Depends(get_db),
           actor: Actor = Depends(require(Permission.INTEGRATIONS_MANAGE)),
           _: Actor = Depends(require_step_up(high_risk=True))):
    cred, raw = svc.rotate_credential(db, actor, client_id, body.grace_hours, client_ip(request))
    return {"credential_id": cred.id, "key_id": cred.key_id, "api_key": raw,
            "api_key_notice": "Store this key now. It will not be shown again."}


@router.post("/clients/{client_id}/credentials/{credential_id}/revoke")
def revoke_credential(client_id: str, credential_id: str, request: Request, db: Session = Depends(get_db),
                      actor: Actor = Depends(require(Permission.INTEGRATIONS_MANAGE))):
    svc.revoke_credential(db, actor, client_id, credential_id, client_ip(request))
    return svc.client_view(db, svc.load_client(db, actor, client_id), detail=True)


@router.post("/clients/{client_id}/revoke")
def revoke_client(client_id: str, request: Request, db: Session = Depends(get_db),
                  actor: Actor = Depends(require(Permission.INTEGRATIONS_MANAGE))):
    return svc.client_view(db, svc.revoke_client(db, actor, client_id, client_ip(request)), detail=True)


@router.post("/clients/{client_id}/webhooks", status_code=201)
def create_webhook(client_id: str, body: WebhookCreate, request: Request, db: Session = Depends(get_db),
                   actor: Actor = Depends(require(Permission.INTEGRATIONS_MANAGE))):
    ep, secret = svc.create_webhook(db, actor, client_id, body.url, body.events, client_ip(request))
    return {"id": ep.id, "url": ep.url, "events": ep.events.split(","), "status": ep.status, "secret": secret,
            "secret_notice": "Store this signing secret now. It will not be shown again."}


@router.post("/webhooks/{webhook_id}/rotate-secret")
def rotate_webhook_secret(webhook_id: str, request: Request, db: Session = Depends(get_db),
                          actor: Actor = Depends(require(Permission.INTEGRATIONS_MANAGE))):
    return {"secret": svc.rotate_webhook_secret(db, actor, webhook_id, client_ip(request))}


@router.post("/webhooks/{webhook_id}/disable")
def disable_webhook(webhook_id: str, db: Session = Depends(get_db),
                    actor: Actor = Depends(require(Permission.INTEGRATIONS_MANAGE))):
    ep = svc.load_webhook(db, actor, webhook_id)
    ep.status = "DISABLED"
    db.commit()
    return {"id": ep.id, "status": ep.status}


@router.post("/webhooks/{webhook_id}/test")
def test_webhook(webhook_id: str, db: Session = Depends(get_db),
                 actor: Actor = Depends(require(Permission.INTEGRATIONS_MANAGE))):
    d = svc.send_ping(db, actor, webhook_id)
    return {"id": d.id, "status": d.status, "attempts": d.attempts, "last_status_code": d.last_status_code,
            "last_error": d.last_error}


@router.post("/deliveries/{delivery_id}/retry")
def retry_delivery(delivery_id: str, db: Session = Depends(get_db),
                   actor: Actor = Depends(require(Permission.INTEGRATIONS_MANAGE))):
    d = (db.query(WebhookDelivery)
         .filter(WebhookDelivery.id == delivery_id, WebhookDelivery.organization_id == actor.org_id).first())
    if d is None:
        raise ApiError(404, "NOT_FOUND", "Delivery not found")
    if d.status == "DELIVERED":
        raise ApiError(409, "ALREADY_DELIVERED", "This delivery already succeeded")
    d.status = "PENDING"
    svc.deliver(db, d)
    return {"id": d.id, "status": d.status, "attempts": d.attempts, "last_status_code": d.last_status_code,
            "last_error": d.last_error}


# ── client API ──────────────────────────────────────────────────────────────

@router.post("/sessions", status_code=201)
def api_create_session(body: ApiSessionCreate, db: Session = Depends(get_db),
                       principal: svc.ApiPrincipal = Depends(api_principal),
                       idempotency_key: Optional[str] = Header(None, max_length=128)):
    principal.require("sessions:write")
    payload = body.model_dump()
    replay = svc.idempotent(db, principal, idempotency_key, payload)
    if replay is not None:
        return JSONResponse(status_code=replay[0], content=replay[1], headers={"Idempotent-Replayed": "true"})
    if not body.subject_user_id and not body.subject_email:
        raise ApiError(422, "SUBJECT_REQUIRED", "Provide subject_user_id or subject_email")
    out = svc.create_session(db, principal, body.subject_user_id, body.subject_email, body.external_ref,
                             body.expires_in)
    from fastapi.encoders import jsonable_encoder
    out = jsonable_encoder(out)
    svc.remember(db, principal, idempotency_key, payload, 201, out)
    return out


@router.get("/sessions/{verification_id}")
def api_get_session(verification_id: str, db: Session = Depends(get_db),
                    principal: svc.ApiPrincipal = Depends(api_principal)):
    principal.require("sessions:read")
    p = svc.load_api_session(db, principal, verification_id)
    out = svc.session_view(db, p)
    out["assertion"] = svc.assertion_for(db, p) if p.status == "COMPLETED" else None
    return out


@router.post("/assertions/introspect")
def api_introspect(body: IntrospectBody, db: Session = Depends(get_db),
                   principal: svc.ApiPrincipal = Depends(api_principal)):
    principal.require("assertions:introspect")
    return svc.introspect(db, principal.client, body.assertion)
