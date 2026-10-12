"""Biometric enrolment router."""
from __future__ import annotations
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.user import User
from backend.models.biometric import BiometricEnrolment, ConsentRecord
from backend.schemas.enrolment import ConsentRequest, EnrolmentResponse, EnrolmentStatus
from backend.services.biometric_service import detect_face, assess_quality, generate_embedding, encrypt_embedding
from backend.services.audit_service import log_security_event
from backend.routers.auth import get_current_user, require_admin

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/enrolments", tags=["enrolments"])


@router.post("/{user_id}/consent")
def record_consent(
    user_id: str,
    data: ConsentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    consent = ConsentRecord(
        user_id=user_id,
        consent_given=data.consent_given,
        consent_text=data.consent_text,
        recorded_at=datetime.utcnow(),
    )
    db.add(consent)

    enrolment = db.query(BiometricEnrolment).filter(BiometricEnrolment.user_id == user_id).first()
    if not enrolment:
        enrolment = BiometricEnrolment(user_id=user_id)
        db.add(enrolment)

    enrolment.consent_given = data.consent_given
    enrolment.consent_timestamp = datetime.utcnow() if data.consent_given else None
    db.commit()

    log_security_event(db, "CONSENT_RECORDED", f"Consent {data.consent_given} for user {user_id}", "INFO", user_id=current_user.id)
    return {"message": "Consent recorded", "consent_given": data.consent_given}


@router.post("/{user_id}/enrol")
async def enrol_user(
    user_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    enrolment = db.query(BiometricEnrolment).filter(BiometricEnrolment.user_id == user_id).first()
    if not enrolment or not enrolment.consent_given:
        raise HTTPException(status_code=400, detail="Consent not given")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    # Detect face
    detection = detect_face(image_bytes)
    if not detection.face_detected:
        raise HTTPException(status_code=400, detail="No face detected in image")
    if detection.face_count > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected")

    # Assess quality
    quality = assess_quality(image_bytes)
    if not quality.is_acceptable:
        raise HTTPException(status_code=400, detail=f"Image quality too low: {quality.message}")

    # Generate embedding
    try:
        embedding = generate_embedding(image_bytes)
    except Exception as e:
        log.warning("Embedding generation failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Could not generate face embedding: {str(e)}")

    # Encrypt and store
    encrypted = encrypt_embedding(embedding)
    enrolment.embedding_encrypted = encrypted
    enrolment.quality_score = quality.quality_score
    enrolment.enrolled_at = datetime.utcnow()
    enrolment.is_active = True
    enrolment.model_name = "ArcFace"
    db.commit()

    log_security_event(db, "ENROLMENT_COMPLETE", f"User {user_id} enrolled", "INFO", user_id=current_user.id)

    return {
        "message": "Enrolment successful",
        "quality_score": quality.quality_score,
        "enrolled_at": enrolment.enrolled_at.isoformat(),
    }


@router.get("/{user_id}", response_model=EnrolmentStatus)
def get_enrolment_status(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    enrolment = db.query(BiometricEnrolment).filter(BiometricEnrolment.user_id == user_id).first()
    if not enrolment:
        return EnrolmentStatus(
            user_id=user_id,
            is_enrolled=False,
            consent_given=False,
            quality_score=0.0,
        )
    return EnrolmentStatus(
        user_id=user_id,
        is_enrolled=enrolment.enrolled_at is not None and enrolment.is_active,
        consent_given=enrolment.consent_given,
        quality_score=enrolment.quality_score,
        enrolled_at=enrolment.enrolled_at,
        model_name=enrolment.model_name if enrolment.enrolled_at else None,
    )


@router.delete("/{user_id}", status_code=204)
def revoke_enrolment(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    enrolment = db.query(BiometricEnrolment).filter(BiometricEnrolment.user_id == user_id).first()
    if not enrolment:
        raise HTTPException(status_code=404, detail="No enrolment found")
    enrolment.embedding_encrypted = None
    enrolment.is_active = False
    enrolment.enrolled_at = None
    db.commit()
    log_security_event(db, "ENROLMENT_REVOKED", f"Enrolment revoked for user {user_id}", "WARNING", user_id=current_user.id)
