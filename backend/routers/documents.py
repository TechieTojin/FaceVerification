"""Identity-document portrait matching."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.deps import Actor, client_ip, require
from backend.core.permissions import Permission
from backend.database import get_db
from backend.models.documents import DocumentCheck
from backend.models.verification import SessionPurpose
from backend.services import capture_service, document_service

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


class CreateCheck(BaseModel):
    subject_label: str = Field("", max_length=200)
    document_type: str = Field("ID_CARD", max_length=24)
    consent_given: bool
    target_user_id: Optional[str] = None


@router.get("/types")
def types(_: Actor = Depends(require(Permission.DOCUMENT_RUN))):
    provider = document_service.get_authenticity_provider()
    return {"document_types": list(document_service.DOCUMENT_TYPES),
            "authenticity_provider": {"name": provider.name, "available": provider.available}}


@router.post("", status_code=201)
def create(body: CreateCheck, request: Request, db: Session = Depends(get_db),
           actor: Actor = Depends(require(Permission.DOCUMENT_RUN))):
    doc = document_service.create_check(db, actor, body.subject_label, body.document_type, body.consent_given,
                                        body.target_user_id, client_ip(request))
    return document_service.view(doc)


@router.get("")
def list_checks(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), db: Session = Depends(get_db),
                actor: Actor = Depends(require(Permission.DOCUMENT_RUN))):
    q = db.query(DocumentCheck).filter(DocumentCheck.organization_id == actor.org_id)
    if not actor.can(Permission.HISTORY_READ_ALL):
        q = q.filter(DocumentCheck.created_by_user_id == actor.id)
    total = q.count()
    rows = q.order_by(DocumentCheck.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "checks": [document_service.view(d) for d in rows]}


@router.get("/{check_id}")
def get_check(check_id: str, db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.DOCUMENT_RUN))):
    return document_service.view(document_service.get_owned(db, actor, check_id))


@router.post("/{check_id}/image")
async def upload(check_id: str, request: Request, file: UploadFile = File(...),
                 actor: Actor = Depends(require(Permission.DOCUMENT_RUN))):
    data = await file.read(8 * 1024 * 1024 + 1)
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(_upload_sync, check_id, actor, data, client_ip(request))


def _upload_sync(check_id: str, actor: Actor, data: bytes, ip: str) -> dict:
    from backend.database import SessionLocal
    db = SessionLocal()
    try:
        return document_service.view(document_service.upload_document(db, actor, check_id, data, ip))
    finally:
        db.close()


@router.post("/{check_id}/session", status_code=201)
def start_live_capture(check_id: str, request: Request, db: Session = Depends(get_db),
                       actor: Actor = Depends(require(Permission.DOCUMENT_RUN))):
    doc = document_service.get_owned(db, actor, check_id)
    if doc.created_by_user_id != actor.id:
        raise HTTPException(status_code=403, detail="Only the operator who created this check can run the live capture")
    created = capture_service.create_session(db, actor, SessionPurpose.DOCUMENT, None, client_ip(request), document=doc)
    view = capture_service.session_view(db, created.session, actor, state=created.state)
    view["frame_token"] = created.frame_token
    return view


@router.delete("/{check_id}")
def delete_check(check_id: str, request: Request, db: Session = Depends(get_db),
                 actor: Actor = Depends(require(Permission.DOCUMENT_RUN))):
    return document_service.view(document_service.delete_check(db, actor, check_id, client_ip(request)))
