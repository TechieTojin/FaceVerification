"""QR pairing (operator desktop) and the paired-phone endpoints (`/remote`, header `X-Pairing-Secret`)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.deps import Actor, client_ip, require
from backend.core.errors import ApiError
from backend.core.permissions import Permission
from backend.database import get_db
from backend.models.pairing import CapturePairing, OPEN_PAIRING_STATUSES
from backend.services import pairing_service

router = APIRouter(prefix="/api/v1/pairings", tags=["cross-device"])
remote = APIRouter(prefix="/api/v1/remote", tags=["cross-device"])


class CreatePairing(BaseModel):
    target_user_id: str = Field(..., max_length=36)
    camera_id: Optional[str] = Field(None, max_length=36)


class ConfirmBody(BaseModel):
    code: str = Field(..., min_length=4, max_length=12)


class ClaimBody(BaseModel):
    accept: bool


def _with_link(db: Session, p: CapturePairing, raw: str) -> dict:
    return {**pairing_service.view(db, p), "qr_url": pairing_service._url(raw)}


@router.post("", status_code=201)
def create(body: CreatePairing, request: Request, db: Session = Depends(get_db),
           actor: Actor = Depends(require(Permission.VERIFY_RUN))):
    p, raw = pairing_service.create_qr(db, actor, body.target_user_id, body.camera_id, client_ip(request))
    return _with_link(db, p, raw)


@router.get("")
def list_mine(open_only: bool = True, db: Session = Depends(get_db),
              actor: Actor = Depends(require(Permission.VERIFY_RUN))):
    q = db.query(CapturePairing).filter(CapturePairing.organization_id == actor.org_id,
                                        CapturePairing.created_by_user_id == actor.id)
    if open_only:
        q = q.filter(CapturePairing.status.in_(OPEN_PAIRING_STATUSES))
    rows = q.order_by(CapturePairing.created_at.desc()).limit(50).all()
    return {"pairings": [pairing_service.view(db, p) for p in rows]}


@router.get("/{pairing_id}")
def get(pairing_id: str, db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.VERIFY_RUN))):
    return pairing_service.view(db, pairing_service.load_for_operator(db, actor, pairing_id))


@router.post("/{pairing_id}/confirm")
def confirm(pairing_id: str, body: ConfirmBody, request: Request, db: Session = Depends(get_db),
            actor: Actor = Depends(require(Permission.VERIFY_RUN))):
    return pairing_service.view(db, pairing_service.confirm(db, actor, pairing_id, body.code, client_ip(request)))


@router.post("/{pairing_id}/cancel")
def cancel(pairing_id: str, request: Request, db: Session = Depends(get_db),
           actor: Actor = Depends(require(Permission.VERIFY_RUN))):
    p = pairing_service.load_for_operator(db, actor, pairing_id)
    return pairing_service.view(db, pairing_service.cancel(db, p, "OPERATOR_CANCELLED", actor.id, client_ip(request)))


@router.post("/{pairing_id}/regenerate", status_code=201)
def regenerate(pairing_id: str, request: Request, db: Session = Depends(get_db),
               actor: Actor = Depends(require(Permission.VERIFY_RUN))):
    """Replace an unused or expired QR with a fresh one for the same subject."""
    p = pairing_service.load_for_operator(db, actor, pairing_id)
    if p.claimed_at is not None and p.status not in ("EXPIRED", "CANCELLED"):
        raise ApiError(409, "ALREADY_CLAIMED", "A phone already claimed this pairing; cancel it first")
    pairing_service.cancel(db, p, "REGENERATED", actor.id, client_ip(request))
    new, raw = pairing_service.create_qr(db, actor, p.target_user_id, p.camera_id, client_ip(request))
    return _with_link(db, new, raw)


# ── paired phone ────────────────────────────────────────────────────────────

@remote.get("/link/{token}")
def link_info(token: str, db: Session = Depends(get_db)):
    return pairing_service.claim_info(db, token)


@remote.post("/link/{token}/claim")
def claim(token: str, body: ClaimBody, request: Request, db: Session = Depends(get_db)):
    if not body.accept:
        raise ApiError(422, "NOT_ACCEPTED", "Confirm that you are the person named and agree to the check")
    p, secret, code = pairing_service.claim(db, token, client_ip(request), request.headers.get("user-agent", ""))
    return {"pairing_secret": secret, "confirmation_code": code, "state": pairing_service.phone_view(db, p)}


@remote.get("/state")
def state(db: Session = Depends(get_db), x_pairing_secret: str = Header("", max_length=128)):
    return pairing_service.phone_view(db, pairing_service.by_secret(db, x_pairing_secret))


@remote.post("/session", status_code=201)
def start(request: Request, db: Session = Depends(get_db), x_pairing_secret: str = Header("", max_length=128)):
    return pairing_service.start_session(db, x_pairing_secret, client_ip(request))


@remote.post("/cancel")
def phone_cancel(request: Request, db: Session = Depends(get_db), x_pairing_secret: str = Header("", max_length=128)):
    p = pairing_service.by_secret(db, x_pairing_secret)
    return pairing_service.phone_view(db, pairing_service.cancel(db, p, "PHONE_CANCELLED", None, client_ip(request)))
