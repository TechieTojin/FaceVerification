"""Identity-document portrait matching.

Three separate results are recorded and never conflated:
  1. face_match_result     — live face vs. the document portrait (ArcFace + liveness)
  2. authenticity_status   — whether the *document* is genuine; requires an external
                             provider. With none configured it is NOT_PERFORMED.
  3. overall_decision      — IDENTITY_VERIFIED only when both 1 and 2 pass;
                             FACE_MATCH_ONLY when the face matched but authenticity
                             was not established.

Data minimisation: the uploaded document image is decoded in memory and
discarded. Only an encrypted portrait template is stored, and it is purged as
soon as the live comparison finishes (or when retention expires).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Protocol

import cv2
import numpy as np
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models.biometric import ConsentRecord
from backend.models.documents import DocumentCheck
from backend.models.user import User
from backend.models.verification import VerificationSession
from backend.services.audit_service import audit, log_security_event
from backend.services.biometric.imaging import InvalidImageError, decode_image
from backend.services.biometric.templates import encrypt_vectors
from backend.services.inference import InferenceBusy, InferenceUnavailable, runtime
from backend.core.deps import Actor, membership_role
from backend.services.policy_service import get_active_policy

log = logging.getLogger(__name__)

DOCUMENT_TYPES = ("PASSPORT", "ID_CARD", "DRIVER_LICENSE", "RESIDENCE_PERMIT", "OTHER")


class DocumentError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass
class AuthenticityResult:
    status: str            # PASSED | FAILED | NOT_PERFORMED | ERROR
    provider: str
    detail: str


class AuthenticityProvider(Protocol):
    name: str
    available: bool

    def check(self, image_bgr: np.ndarray, document_type: str) -> AuthenticityResult: ...


class NoAuthenticityProvider:
    """No external document-forensics service is configured. Never returns PASSED."""

    name = "none"
    available = False

    def check(self, image_bgr: np.ndarray, document_type: str) -> AuthenticityResult:
        return AuthenticityResult("NOT_PERFORMED", self.name,
                                  "No document authenticity provider is configured; the document was not authenticated.")


def get_authenticity_provider() -> AuthenticityProvider:
    # Extension point: add a concrete provider class and select it with DOCUMENT_AUTH_PROVIDER.
    return NoAuthenticityProvider()


def create_check(db: Session, actor: Actor, subject_label: str, document_type: str, consent_given: bool,
                 target_user_id: Optional[str], ip: str) -> DocumentCheck:
    if not consent_given:
        raise DocumentError(400, "CONSENT_REQUIRED", "The subject must consent to document verification")
    if document_type not in DOCUMENT_TYPES:
        raise DocumentError(422, "INVALID_TYPE", f"document_type must be one of {', '.join(DOCUMENT_TYPES)}")
    if target_user_id and membership_role(db, actor.org_id, target_user_id) is None:
        raise DocumentError(404, "TARGET_NOT_FOUND", "Target user not found")
    policy = get_active_policy(db, actor.org_id)
    now = datetime.utcnow()
    doc = DocumentCheck(organization_id=actor.org_id, created_by_user_id=actor.id, target_user_id=target_user_id,
                        subject_label=(subject_label or "").strip()[:200] or None, document_type=document_type,
                        consent_given=True, consent_at=now, status="AWAITING_DOCUMENT",
                        authenticity_provider=get_authenticity_provider().name,
                        retention_expires_at=now + timedelta(hours=int(policy.retention_documents_hours)))
    db.add(doc)
    db.flush()
    if target_user_id:
        db.add(ConsentRecord(organization_id=actor.org_id, user_id=target_user_id, consent_given=True,
                             purpose="DOCUMENT_VERIFICATION",
                             method="OPERATOR_ATTESTED", actor_user_id=actor.id, ip_address=ip, recorded_at=now,
                             consent_text="Subject consented to identity-document portrait comparison."))
    audit(db, actor.id, "DOCUMENT_CHECK_CREATED", "document_check", doc.id, {"type": document_type}, ip, commit=False,
          org_id=actor.org_id)
    db.commit()
    return doc


def _document_quality(bgr: np.ndarray, face_w_ratio: float) -> tuple[float, list[str]]:
    issues: list[str] = []
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (800, max(1, int(h * 800 / w))), interpolation=cv2.INTER_AREA)
    sharp = min(1.0, float(cv2.Laplacian(small, cv2.CV_64F).var()) / 200.0)
    bright = float(small.mean()) / 255.0
    glare = float((small >= 250).mean())
    if w < 600:
        issues.append("LOW_RESOLUTION")
    if sharp < 0.2:
        issues.append("BLURRY")
    if bright < 0.2:
        issues.append("TOO_DARK")
    if glare > 0.06:
        issues.append("GLARE")
    if face_w_ratio < 0.06:
        issues.append("PORTRAIT_TOO_SMALL")
    score = round(0.5 * sharp + 0.3 * max(0.0, 1 - abs(bright - 0.55) * 2) + 0.2 * (1 - min(1.0, glare * 10)), 3)
    return score, issues


def _process_document(provider, bgr: np.ndarray, auth: AuthenticityProvider, document_type: str):
    analysis = provider.analyze(bgr)
    if not analysis.faces:
        return None, analysis, None, auth.check(bgr, document_type), (0.0, [])
    portrait = max(analysis.faces, key=lambda f: f.bbox[2] * f.bbox[3])
    vec = provider.embed(bgr, portrait)
    return portrait, analysis, vec, auth.check(bgr, document_type), _document_quality(bgr, portrait.bbox[2])


def upload_document(db: Session, actor: Actor, check_id: str, data: bytes, ip: str) -> DocumentCheck:
    doc = get_owned(db, actor, check_id)
    if doc.status not in ("AWAITING_DOCUMENT", "AWAITING_LIVE_CAPTURE"):
        raise DocumentError(409, "CHECK_CLOSED", f"This check is {doc.status}")
    if len(data) > 8 * 1024 * 1024:
        raise DocumentError(413, "TOO_LARGE", "Document image exceeds 8 MB")
    try:
        bgr = decode_image(data, max_side=2000)
    except InvalidImageError as exc:
        raise DocumentError(422, "INVALID_IMAGE", str(exc)) from exc
    auth = get_authenticity_provider()
    try:
        portrait, analysis, vec, auth_res, (score, issues) = runtime.run(
            _process_document, runtime.provider, bgr, auth, doc.document_type, timeout=60.0)
    except (InferenceBusy, InferenceUnavailable) as exc:
        raise DocumentError(503, "INFERENCE_BUSY", "Models are busy or unavailable; retry shortly") from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Document processing failed")
        raise DocumentError(422, "PROCESSING_FAILED", "The document image could not be processed") from exc
    finally:
        del bgr  # the document image is never persisted

    doc.authenticity_status, doc.authenticity_provider, doc.authenticity_detail = auth_res.status, auth_res.provider, auth_res.detail
    if portrait is None:
        doc.portrait_detected = False
        doc.document_quality_issues = "NO_PORTRAIT"
        doc.status = "AWAITING_DOCUMENT"
        db.commit()
        raise DocumentError(422, "NO_PORTRAIT", "No portrait face was found on the document. Capture it flat and well lit.")
    issues = list(issues)
    if len(analysis.faces) > 1:
        issues.append("MULTIPLE_FACES")
    doc.document_quality_score, doc.document_quality_issues = score, ",".join(issues) or None
    blocking = [i for i in issues if i in ("BLURRY", "TOO_DARK", "PORTRAIT_TOO_SMALL", "LOW_RESOLUTION")]
    if blocking:
        doc.portrait_detected = True
        doc.status = "AWAITING_DOCUMENT"
        db.commit()
        raise DocumentError(422, "DOCUMENT_QUALITY", f"Document image quality too low: {', '.join(blocking)}")
    doc.portrait_detected = True
    doc.portrait_template_encrypted = encrypt_vectors([vec], runtime.provider.recognition_model)
    doc.status = "AWAITING_LIVE_CAPTURE"
    audit(db, actor.id, "DOCUMENT_UPLOADED", "document_check", doc.id,
          {"quality": score, "issues": issues, "authenticity": auth_res.status}, ip, commit=False, org_id=actor.org_id)
    db.commit()
    return doc


def get_owned(db: Session, actor: Actor, check_id: str) -> DocumentCheck:
    from backend.core.permissions import Permission
    doc = db.query(DocumentCheck).filter(DocumentCheck.id == check_id, DocumentCheck.organization_id == actor.org_id).first()
    if doc is None or (doc.created_by_user_id != actor.id and not actor.can(Permission.HISTORY_READ_ALL)):
        raise DocumentError(404, "NOT_FOUND", "Document check not found")
    return doc


def complete_from_session(db: Session, s: VerificationSession) -> None:
    doc = db.query(DocumentCheck).filter(DocumentCheck.id == s.document_check_id).first()
    if doc is None or doc.status not in ("AWAITING_LIVE_CAPTURE",):
        return
    now = datetime.utcnow()
    outcome = s.final_outcome
    doc.session_id = s.id
    doc.threshold = s.threshold
    if outcome == "VERIFIED":
        doc.face_match_result, doc.liveness_passed = "MATCH", True
    elif outcome == "NOT_MATCHED":
        doc.face_match_result, doc.liveness_passed = "NO_MATCH", True
    elif outcome == "LIVENESS_FAILED":
        doc.face_match_result, doc.liveness_passed = "NOT_EVALUATED", False
    else:
        doc.face_match_result = "NOT_EVALUATED"
    doc.face_match_distance = s.final_distance
    if doc.face_match_result == "MATCH":
        doc.overall_decision = "IDENTITY_VERIFIED" if doc.authenticity_status == "PASSED" else "FACE_MATCH_ONLY"
    elif doc.face_match_result == "NO_MATCH":
        doc.overall_decision = "NOT_VERIFIED"
    elif doc.liveness_passed is False:
        doc.overall_decision = "LIVENESS_FAILED"
    else:
        doc.overall_decision = "INCOMPLETE"
    doc.status = "COMPLETED"
    doc.completed_at = now
    doc.portrait_template_encrypted = None
    doc.purged_at = now
    log_security_event(db, "DOCUMENT_CHECK_COMPLETED", f"Document check {doc.id[:8]}: {doc.overall_decision}", "INFO",
                       user_id=s.created_by_user_id, metadata={"document_check_id": doc.id}, commit=False,
                       org_id=s.organization_id)


def delete_check(db: Session, actor: Actor, check_id: str, ip: str) -> DocumentCheck:
    doc = get_owned(db, actor, check_id)
    doc.portrait_template_encrypted = None
    doc.purged_at = datetime.utcnow()
    if doc.status != "COMPLETED":
        doc.status = "DELETED"
    audit(db, actor.id, "DOCUMENT_CHECK_DELETED", "document_check", doc.id, None, ip, commit=False, org_id=actor.org_id)
    db.commit()
    return doc


def purge_expired(db: Session) -> int:
    now = datetime.utcnow()
    rows = (db.query(DocumentCheck)
            .filter(DocumentCheck.retention_expires_at < now, DocumentCheck.portrait_template_encrypted.isnot(None)).all())
    for doc in rows:
        doc.portrait_template_encrypted = None
        doc.purged_at = now
        if doc.status in ("AWAITING_DOCUMENT", "AWAITING_LIVE_CAPTURE"):
            doc.status = "EXPIRED"
    return len(rows)


def view(doc: DocumentCheck) -> dict:
    return {
        "id": doc.id, "status": doc.status, "subject_label": doc.subject_label, "target_user_id": doc.target_user_id,
        "document_type": doc.document_type, "created_at": doc.created_at, "completed_at": doc.completed_at,
        "consent_given": doc.consent_given, "document_quality_score": doc.document_quality_score,
        "document_quality_issues": doc.document_quality_issues.split(",") if doc.document_quality_issues else [],
        "portrait_detected": doc.portrait_detected, "portrait_template_stored": doc.portrait_template_encrypted is not None,
        "face_match_result": doc.face_match_result, "face_match_distance": doc.face_match_distance,
        "threshold": doc.threshold, "liveness_passed": doc.liveness_passed,
        "authenticity_status": doc.authenticity_status, "authenticity_provider": doc.authenticity_provider,
        "authenticity_detail": doc.authenticity_detail, "overall_decision": doc.overall_decision,
        "session_id": doc.session_id, "retention_expires_at": doc.retention_expires_at, "purged_at": doc.purged_at,
    }
