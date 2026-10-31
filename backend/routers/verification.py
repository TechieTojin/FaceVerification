"""Capture sessions: create (verify), submit frames, cancel, read."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from typing import Optional

from backend.core.capture_auth import get_capture_actor
from backend.core.deps import Actor, client_ip, require
from backend.core.permissions import Permission
from backend.database import get_db
from backend.models.user import User
from backend.models.verification import SessionPurpose
from backend.services import capture_service

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])

MAX_UPLOAD = 2_100_000  # hard ceiling; the policy limit is enforced per session


class CreateSession(BaseModel):
    target_user_id: str
    camera_id: Optional[str] = None


async def _read_bounded(file: UploadFile, limit: int) -> bytes:
    chunks, total = [], 0
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="Frame too large")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("", status_code=201)
def create(body: CreateSession, request: Request, db: Session = Depends(get_db),
           actor: Actor = Depends(require(Permission.VERIFY_RUN))):
    target = db.query(User).filter(User.id == body.target_user_id).first()
    created = capture_service.create_session(db, actor, SessionPurpose.VERIFY, target, client_ip(request),
                                             camera_id=body.camera_id)
    view = capture_service.session_view(db, created.session, actor, state=created.state)
    view["frame_token"] = created.frame_token
    return view


@router.post("/{session_id}/frames")
async def submit_frame(session_id: str, request: Request, seq: int = Form(..., ge=1), frame_token: str = Form(..., max_length=128),
                       file: UploadFile = File(...), actor: Actor = Depends(get_capture_actor)):
    data = await _read_bounded(file, MAX_UPLOAD)
    return await submit_frame_async(actor, session_id, seq, frame_token, data, client_ip(request))


async def read_frame(file: UploadFile) -> bytes:
    return await _read_bounded(file, MAX_UPLOAD)


async def submit_frame_async(actor: Actor, session_id: str, seq: int, token: str, data: bytes, ip: str) -> dict:
    """Run frame submission (blocking DB + inference) off the event loop with its own DB session."""
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(_submit_sync, actor, session_id, seq, token, data, ip)


def _submit_sync(actor: Actor, session_id: str, seq: int, token: str, data: bytes, ip: str) -> dict:
    from backend.database import SessionLocal
    db = SessionLocal()
    try:
        return capture_service.submit_frame(db, actor, session_id, seq, token, data, ip)
    finally:
        db.close()


@router.post("/{session_id}/resync")
def resync(session_id: str, request: Request, db: Session = Depends(get_db), actor: Actor = Depends(get_capture_actor)):
    return capture_service.resync_session(db, actor, session_id, client_ip(request))


@router.post("/{session_id}/cancel")
def cancel(session_id: str, request: Request, db: Session = Depends(get_db), actor: Actor = Depends(get_capture_actor)):
    return capture_service.cancel_session(db, actor, session_id, client_ip(request))


@router.get("/{session_id}")
def get(session_id: str, request: Request, db: Session = Depends(get_db), actor: Actor = Depends(get_capture_actor)):
    s = capture_service.load_owned_session(db, actor, session_id, client_ip(request), allow_admin_read=True)
    if capture_service.expire_if_needed(db, s, datetime.utcnow()):
        db.commit()
    return capture_service.session_view(db, s, actor)
