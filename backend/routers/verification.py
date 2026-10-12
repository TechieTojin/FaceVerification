"""Verification session and frame submission router."""
from __future__ import annotations
import logging
import time
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.config import settings
from backend.models.user import User
from backend.models.biometric import BiometricEnrolment
from backend.models.verification import VerificationSession, VerificationEvent, OutcomeEnum
from backend.schemas.verification import SessionCreateRequest, SessionResponse, FrameResult, PaginatedEvents, VerificationEventResponse
from backend.services.biometric_service import detect_face, check_liveness, generate_embedding, decrypt_embedding, compare_embeddings, image_bytes_to_array
from backend.services.audit_service import log_security_event
from backend.routers.auth import get_current_user

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/verification", tags=["verification"])


@router.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session(
    data: SessionCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    target_user = db.query(User).filter(User.id == data.target_user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="Target user not found")

    enrolment = db.query(BiometricEnrolment).filter(
        BiometricEnrolment.user_id == data.target_user_id,
        BiometricEnrolment.is_active == True,
    ).first()
    if not enrolment or not enrolment.enrolled_at:
        raise HTTPException(status_code=400, detail="User is not enrolled")

    session = VerificationSession(
        target_user_id=data.target_user_id,
        created_by_user_id=current_user.id,
        expires_at=datetime.utcnow() + timedelta(seconds=settings.session_timeout_seconds),
        is_active=True,
        frame_count=0,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return SessionResponse(
        session_id=session.id,
        target_user_id=session.target_user_id,
        expires_at=session.expires_at,
        is_active=session.is_active,
        frame_count=session.frame_count,
    )


@router.post("/sessions/{session_id}/submit-frame", response_model=FrameResult)
async def submit_frame(
    session_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    t0 = time.monotonic()

    # Validate session
    session = db.query(VerificationSession).filter(VerificationSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is no longer active")
    if datetime.utcnow() > session.expires_at:
        session.is_active = False
        session.final_outcome = OutcomeEnum.TIMEOUT
        db.commit()
        raise HTTPException(status_code=400, detail="Session expired")

    attempts_used = session.frame_count
    max_attempts = settings.max_attempts_per_session

    if attempts_used >= max_attempts:
        session.is_active = False
        session.final_outcome = OutcomeEnum.LOCKOUT
        db.commit()
        return FrameResult(
            session_id=session_id,
            outcome=OutcomeEnum.LOCKOUT,
            action="lockout",
            explanation="Maximum attempts exceeded. Session locked.",
            face_count=0,
            inference_ms=0.0,
            attempts_remaining=0,
        )

    session.frame_count += 1
    db.commit()

    image_bytes = await file.read()
    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Frame too large (max 5MB)")

    # Fail-closed processing
    try:
        outcome, action, explanation, distance, threshold, liveness_score, face_count, is_liveness_real = _process_frame(
            image_bytes, session.target_user_id, db
        )
    except Exception as e:
        log.error("Frame processing error: %s", e)
        outcome = OutcomeEnum.ERROR
        action = "retry"
        explanation = "Processing error. Please try again."
        distance = None
        threshold = None
        liveness_score = None
        face_count = 0
        is_liveness_real = None

    inference_ms = (time.monotonic() - t0) * 1000
    attempts_remaining = max_attempts - session.frame_count

    # Record event
    event = VerificationEvent(
        session_id=session_id,
        user_id=session.target_user_id,
        outcome=outcome,
        distance=distance,
        threshold=threshold,
        liveness_score=liveness_score,
        face_count=face_count,
        inference_ms=inference_ms,
        explanation=explanation,
        policy_version="v1.0",
        is_liveness_real=is_liveness_real,
    )
    db.add(event)

    if outcome == OutcomeEnum.VERIFIED or action in ("reject", "lockout"):
        if outcome == OutcomeEnum.VERIFIED:
            session.final_outcome = OutcomeEnum.VERIFIED
            session.is_active = False
        elif action == "lockout":
            session.final_outcome = OutcomeEnum.LOCKOUT
            session.is_active = False
        else:
            session.final_outcome = outcome
            session.is_active = False
        session.completed_at = datetime.utcnow()

    db.commit()

    if outcome == OutcomeEnum.SPOOF_DETECTED:
        log_security_event(db, "SPOOF_DETECTED", f"Spoof detected for user {session.target_user_id}", "HIGH", user_id=session.target_user_id)

    return FrameResult(
        session_id=session_id,
        outcome=outcome,
        action=action,
        explanation=explanation,
        distance=distance,
        threshold=threshold,
        liveness_score=liveness_score,
        face_count=face_count,
        inference_ms=round(inference_ms, 1),
        attempts_remaining=max(0, attempts_remaining),
        is_liveness_real=is_liveness_real,
    )


def _process_frame(image_bytes: bytes, target_user_id: str, db: Session):
    """Run the full verification pipeline. Returns tuple of result fields."""
    # Step 1: Quality / detect
    detection = detect_face(image_bytes)
    face_count = detection.face_count

    if face_count == 0:
        return OutcomeEnum.NO_FACE, "retry", "No face detected. Position your face in frame.", None, None, None, 0, None

    if face_count > 1:
        return OutcomeEnum.MULTIPLE_FACES, "reject", "Multiple faces detected. Ensure only one face is in frame.", None, None, None, face_count, None

    # Step 2: Liveness
    is_liveness_real = None
    liveness_score = None
    if settings.enable_liveness:
        liveness = check_liveness(image_bytes)
        is_liveness_real = liveness["is_real"]
        liveness_score = liveness["score"]
        if not is_liveness_real:
            return OutcomeEnum.SPOOF_DETECTED, "reject", "Liveness check failed. Please use a real face.", None, None, liveness_score, face_count, is_liveness_real

    # Step 3: Identity verification
    enrolment = db.query(BiometricEnrolment).filter(
        BiometricEnrolment.user_id == target_user_id,
        BiometricEnrolment.is_active == True,
    ).first()
    if not enrolment or not enrolment.embedding_encrypted:
        return OutcomeEnum.ERROR, "retry", "No enrolled face found.", None, None, liveness_score, face_count, is_liveness_real

    # Generate live embedding
    try:
        live_embedding = generate_embedding(image_bytes)
    except Exception as e:
        return OutcomeEnum.NO_FACE, "retry", "Could not extract face features. Try again.", None, None, liveness_score, face_count, is_liveness_real

    # Decrypt stored embedding
    stored_embedding = decrypt_embedding(enrolment.embedding_encrypted)

    # Adjust embedding sizes if needed
    min_len = min(len(live_embedding), len(stored_embedding))
    distance = compare_embeddings(live_embedding[:min_len], stored_embedding[:min_len])
    threshold = settings.max_face_distance

    if distance <= threshold:
        return OutcomeEnum.VERIFIED, "verified", "Identity verified. Welcome!", distance, threshold, liveness_score, face_count, is_liveness_real
    else:
        return OutcomeEnum.REJECTED, "reject", "Face does not match enrolled profile.", distance, threshold, liveness_score, face_count, is_liveness_real


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = db.query(VerificationSession).filter(VerificationSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        session_id=session.id,
        target_user_id=session.target_user_id,
        expires_at=session.expires_at,
        is_active=session.is_active,
        frame_count=session.frame_count,
    )


@router.get("/events", response_model=PaginatedEvents)
def list_events(
    user_id: Optional[str] = None,
    outcome: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(VerificationEvent)
    if user_id:
        q = q.filter(VerificationEvent.user_id == user_id)
    if outcome:
        q = q.filter(VerificationEvent.outcome == outcome)
    total = q.count()
    events = q.order_by(VerificationEvent.timestamp.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return PaginatedEvents(
        total=total,
        page=page,
        page_size=page_size,
        events=[VerificationEventResponse(
            id=e.id,
            session_id=e.session_id,
            user_id=e.user_id,
            timestamp=e.timestamp,
            outcome=e.outcome,
            distance=e.distance,
            threshold=e.threshold,
            liveness_score=e.liveness_score,
            face_count=e.face_count,
            inference_ms=e.inference_ms,
            explanation=e.explanation,
        ) for e in events],
    )
