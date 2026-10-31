"""Capture sessions (VERIFY / ENROL / DOCUMENT) — server-side state machine.

Frame protocol (per session):
  * The server issues a one-time `frame_token` and expects `seq = frame_seq + 1`.
  * A frame is *claimed* with one atomic UPDATE that requires: session active,
    matching token hash and sequence, not already in flight, frame budget left.
    That gives exactly one in-flight frame per session, atomic frame counting
    and rejection of stale/duplicated submissions — without row locks.
  * The claimed frame runs on the bounded inference runtime; the result is
    applied with `WHERE is_active` so a concurrent cancel always wins.
  * Exact byte-identical frames seen in the replay window are rejected.

Ownership: a session is driven only by its owner — `(owner_kind, owner_ref)`:
USER (the operator who created it), PAIRING (a phone holding a confirmed pairing
secret, or a hosted integration link), LOGIN (a passkey sign-in completing its
face check). Sessions belong to one organisation; a session of another
organisation is reported as not found and raises a cross-tenant alert.

What this does NOT prove: that pixels came from a physical camera. A virtual
camera or modified browser can feed arbitrary video; active challenges and
passive anti-spoofing raise the cost of such attacks, they do not eliminate it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.config import settings
from backend.core.events import bus
from backend.core.deps import Actor, membership_role
from backend.core.permissions import Permission
from backend.models.biometric import BiometricEnrolment, EnrolmentStatus
from backend.models.devices import Camera
from backend.models.documents import DocumentCheck
from backend.models.user import User
from backend.models.verification import (
    TERMINAL_OUTCOMES, FrameDigest, LivenessChallenge, OutcomeEnum, SessionPurpose,
    VerificationEvent, VerificationSession,
)
from backend.services import alert_service, enrolment_service
from backend.services.audit_service import log_security_event
from backend.services.biometric.analysis import FaceObservation, FrameAnalysis, PassiveResult
from backend.services.biometric.engine import Engine, EngineState, FrameDecision, StepEvent, step_view
from backend.services.biometric.imaging import InvalidImageError, decode_image
from backend.services.biometric.templates import (
    TemplateError, compare_to_template, decrypt_vectors, encrypt_vectors,
)
from backend.services.inference import InferenceBusy, InferenceUnavailable, runtime
from backend.services.policy_service import Policy, get_active_policy

log = logging.getLogger(__name__)

IN_FLIGHT_STALE_S = 45
MAX_CONSECUTIVE_ERRORS = 3
MAX_REPLAYED_FRAMES = 5


class CaptureError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: Optional[dict] = None,
                 payload: Optional[dict] = None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers, self.payload = status, code, message, headers, payload


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _new_token() -> tuple[str, str]:
    raw = secrets.token_urlsafe(24)
    return raw, _hash(raw)


def _policy_of(session: VerificationSession) -> Policy:
    return Policy.from_json(session.policy_snapshot)


# ── reference templates ─────────────────────────────────────────────────────

USABILITY_MESSAGES = {
    EnrolmentStatus.NOT_ENROLLED: ("NOT_ENROLLED", "This user is not enrolled"),
    EnrolmentStatus.REVOKED: ("ENROLMENT_REVOKED", "This user's enrolment was revoked"),
    EnrolmentStatus.WITHDRAWN: ("CONSENT_WITHDRAWN", "This user withdrew biometric consent"),
    EnrolmentStatus.REENROLMENT_REQUIRED: ("REENROLMENT_REQUIRED",
                                           "This user's template predates v3 and must be re-enrolled"),
}


def check_verifiable(enr: Optional[BiometricEnrolment]) -> None:
    if enr is None:
        raise CaptureError(409, *USABILITY_MESSAGES[EnrolmentStatus.NOT_ENROLLED])
    if not enr.consent_given:
        raise CaptureError(409, "CONSENT_WITHDRAWN", "Biometric consent is not in place for this user")
    if enr.status != EnrolmentStatus.ACTIVE or enr.embedding_encrypted is None:
        code, msg = USABILITY_MESSAGES.get(enr.status, ("NOT_ENROLLED", "This user is not enrolled"))
        raise CaptureError(409, code, msg)
    if enr.template_version < 2:
        raise CaptureError(409, *USABILITY_MESSAGES[EnrolmentStatus.REENROLMENT_REQUIRED])


# ── session creation ────────────────────────────────────────────────────────

@dataclass
class CreatedSession:
    session: VerificationSession
    frame_token: str
    state: EngineState


def _close_superseded(db: Session, actor: Actor, now: datetime) -> None:
    for s in (db.query(VerificationSession)
              .filter(VerificationSession.organization_id == actor.org_id,
                      VerificationSession.owner_kind == actor.kind, VerificationSession.owner_ref == actor.ref,
                      VerificationSession.is_active == True)  # noqa: E712
              .all()):
        _close(db, s, OutcomeEnum.SESSION_CANCELLED, "SUPERSEDED", now)


def require_encryption_key() -> None:
    try:
        settings.get_fernet_key()
    except RuntimeError as exc:
        log.error("%s", exc)
        raise CaptureError(503, "ENCRYPTION_KEY_MISSING",
                           "Biometric template encryption is not configured on the server") from exc


def _camera(db: Session, actor: Actor, camera_id: Optional[str]) -> Optional[Camera]:
    if not camera_id:
        return None
    cam = db.query(Camera).filter(Camera.id == camera_id, Camera.organization_id == actor.org_id).first()
    if cam is None:
        raise CaptureError(404, "CAMERA_NOT_FOUND", "Camera not found")
    if cam.status != "ACTIVE":
        raise CaptureError(409, "CAMERA_DISABLED", "This camera has been disabled by an administrator")
    return cam


def create_session(db: Session, actor: Actor, purpose: SessionPurpose, target: Optional[User], ip: str,
                   document: Optional[DocumentCheck] = None, *, channel: str = "DESKTOP",
                   camera_id: Optional[str] = None, pairing_id: Optional[str] = None,
                   created_by_user_id: Optional[str] = None) -> CreatedSession:
    require_encryption_key()
    policy = get_active_policy(db, actor.org_id)
    now = datetime.utcnow()

    recent = 0
    if actor.kind == "USER":
        recent = (db.query(VerificationSession)
                  .filter(VerificationSession.organization_id == actor.org_id,
                          VerificationSession.created_by_user_id == actor.id,
                          VerificationSession.created_at >= now - timedelta(minutes=10)).count())
        if recent >= policy.operator_session_rate_limit:
            alert_service.raise_alert(db, "SESSION_RATE_LIMITED", "HIGH", "user", actor.id,
                                      f"{actor.email} exceeded {policy.operator_session_rate_limit} sessions / 10 min",
                                      window_minutes=10, commit=False, org_id=actor.org_id)
            db.commit()
            raise CaptureError(429, "SESSION_RATE_LIMITED", "Too many sessions created recently. Wait and retry.",
                               headers={"Retry-After": "120"})

    camera = _camera(db, actor, camera_id)
    if purpose == SessionPurpose.VERIFY:
        if target is None or not target.is_active or membership_role(db, actor.org_id, target.id) is None:
            raise CaptureError(404, "TARGET_NOT_FOUND", "Target user not found or disabled")
        enr = enrolment_service.get_enrolment(db, actor.org_id, target.id)
        check_verifiable(enr)
    elif purpose == SessionPurpose.ENROL:
        if actor.kind != "USER":
            raise CaptureError(403, "ENROL_NOT_PERMITTED", "Enrolment requires an operator session")
        if target is None:
            raise CaptureError(404, "TARGET_NOT_FOUND", "Target user not found")
        try:
            enrolment_service.ensure_capture_allowed(db, actor, target, ip)
        except enrolment_service.EnrolmentDenied as exc:
            raise CaptureError(exc.status, exc.code, exc.message) from exc
    elif purpose == SessionPurpose.DOCUMENT:
        if document is None or document.status != "AWAITING_LIVE_CAPTURE" or document.portrait_template_encrypted is None:
            raise CaptureError(409, "DOCUMENT_NOT_READY", "Upload a document with a detectable portrait first")

    _close_superseded(db, actor, now)
    engine = Engine(policy, purpose.value)
    state, events = engine.start(now)
    token, token_hash = _new_token()
    session = VerificationSession(
        organization_id=actor.org_id, owner_kind=actor.kind, owner_ref=actor.ref, channel=channel,
        camera_id=camera.id if camera else None, pairing_id=pairing_id,
        purpose=purpose.value, target_user_id=target.id if target else None,
        created_by_user_id=created_by_user_id or actor.id,
        document_check_id=document.id if document else None, created_at=now,
        expires_at=now + timedelta(seconds=int(policy.session_timeout_seconds)), is_active=True,
        frame_count=0, attempts_used=0, frame_seq=0, frame_token_hash=token_hash, in_flight=False,
        last_activity_at=now, policy_version=policy.version, policy_snapshot=policy.to_json(),
        challenge_round=0, challenge_index=0, step_started_at=now, state_json=state.to_json(),
    )
    db.add(session)
    db.flush()
    _apply_step_events(db, session, events, now)
    if actor.kind == "USER":
        alert_service.evaluate_session_creation(db, actor.org_id, actor.id, recent + 1, policy)
    if camera is not None:
        camera.last_heartbeat_at = now
        camera.last_heartbeat_by = actor.email[:64]
    db.commit()
    bus.publish("sessions", "session_created", {"id": session.id, "purpose": session.purpose,
                                                "channel": session.channel}, org_id=actor.org_id)
    return CreatedSession(session=session, frame_token=token, state=state)


# ── frame submission ────────────────────────────────────────────────────────

class _Ops:
    """Engine callbacks that execute on the inference worker thread."""

    def __init__(self, provider, bgr: np.ndarray, reference_blob: Optional[bytes]):
        self.provider, self.bgr, self.reference_blob = provider, bgr, reference_blob
        self._reference = None
        self.timings: dict[str, float] = {}

    def passive(self, face: FaceObservation) -> PassiveResult:
        res = self.provider.passive_liveness(self.bgr, face)
        self.timings["passive_ms"] = self.timings.get("passive_ms", 0.0) + res.ms
        return res

    def embed(self, face: FaceObservation) -> np.ndarray:
        import time
        t0 = time.perf_counter()
        try:
            return self.provider.embed(self.bgr, face)
        finally:
            self.timings["embed_ms"] = self.timings.get("embed_ms", 0.0) + (time.perf_counter() - t0) * 1000

    def distance(self, vec: np.ndarray) -> float:
        if self.reference_blob is None:
            raise TemplateError("No reference template is available")
        if self._reference is None:
            self._reference = decrypt_vectors(self.reference_blob)
        return compare_to_template(vec, self._reference, self.provider.recognition_model)


def _run_engine(provider, bgr: np.ndarray, engine: Engine, state: EngineState, now: datetime,
                reference_blob: Optional[bytes]) -> tuple[FrameAnalysis, FrameDecision, dict[str, float]]:
    analysis = provider.analyze(bgr)
    ops = _Ops(provider, bgr, reference_blob)
    decision = engine.process(state, analysis, now, ops)
    ops.timings["analyze_ms"] = analysis.analyze_ms
    return analysis, decision, ops.timings


def is_owner(actor: Actor, s: VerificationSession) -> bool:
    return s.owner_kind == actor.kind and s.owner_ref == actor.ref


def load_owned_session(db: Session, actor: Actor, session_id: str, ip: str, allow_admin_read: bool = False) -> VerificationSession:
    s = db.query(VerificationSession).filter(VerificationSession.id == session_id).first()
    if s is None:
        raise CaptureError(404, "SESSION_NOT_FOUND", "Session not found")
    if s.organization_id != actor.org_id:
        log_security_event(db, "CROSS_TENANT_ACCESS", f"{actor.email} referenced a session of another organisation",
                           "HIGH", user_id=actor.id, ip_address=ip, metadata={"session_id": session_id[:36]},
                           commit=False, org_id=actor.org_id)
        alert_service.raise_alert(db, "CROSS_TENANT_ACCESS", "HIGH", "user", actor.ref,
                                  f"{actor.email} referenced a capture session of another organisation",
                                  commit=False, org_id=actor.org_id)
        db.commit()
        raise CaptureError(404, "SESSION_NOT_FOUND", "Session not found")
    if not is_owner(actor, s) and not (allow_admin_read and actor.can(Permission.HISTORY_READ_ALL)):
        log_security_event(db, "SESSION_ACCESS_VIOLATION", f"{actor.email} accessed session {session_id[:8]} owned by another principal",
                           "HIGH", user_id=actor.id, ip_address=ip, metadata={"session_id": session_id}, commit=False,
                           org_id=actor.org_id)
        alert_service.raise_alert(db, "SESSION_ACCESS_VIOLATION", "HIGH", "user", actor.ref,
                                  f"{actor.email} tried to use a capture session they do not own", commit=False,
                                  org_id=actor.org_id)
        db.commit()
        raise CaptureError(404, "SESSION_NOT_FOUND", "Session not found")
    return s


def expire_if_needed(db: Session, s: VerificationSession, now: datetime) -> bool:
    if not s.is_active:
        return False
    policy = _policy_of(s)
    reason = None
    if now > s.expires_at:
        reason = "SESSION_TIMEOUT"
    elif s.last_activity_at and (now - s.last_activity_at).total_seconds() > policy.inactivity_timeout_seconds and not s.in_flight:
        reason = "INACTIVITY_TIMEOUT"
    if reason:
        _close(db, s, OutcomeEnum.SESSION_EXPIRED, reason, now)
        return True
    return False


def submit_frame(db: Session, actor: Actor, session_id: str, seq: int, frame_token: str, data: bytes,
                 ip: str) -> dict[str, Any]:
    now = datetime.utcnow()
    s = load_owned_session(db, actor, session_id, ip)
    require_encryption_key()
    if expire_if_needed(db, s, now):
        db.commit()
        return session_view(db, s, actor, frame=None)
    if not s.is_active:
        raise CaptureError(409, "SESSION_CLOSED", "This session has already ended", payload=session_view(db, s, actor))
    policy = _policy_of(s)

    if s.frame_count >= policy.max_frames_per_session:
        _close(db, s, OutcomeEnum.ATTEMPTS_EXHAUSTED, "FRAME_LIMIT", now)
        db.commit()
        return session_view(db, s, actor, frame=None)
    if len(data) > policy.max_frame_bytes:
        raise CaptureError(413, "FRAME_TOO_LARGE", f"Frame exceeds {policy.max_frame_bytes} bytes")
    try:
        bgr = decode_image(data, max_side=1280)
    except InvalidImageError as exc:
        raise CaptureError(422, "INVALID_IMAGE", str(exc)) from exc

    digest = hashlib.sha256(data).hexdigest()
    replay_since = now - timedelta(hours=int(policy.retention_frame_digest_hours))
    if db.query(FrameDigest).filter(FrameDigest.digest == digest, FrameDigest.created_at >= replay_since).first():
        log_security_event(db, "FRAME_REPLAY_DETECTED", f"Replayed frame on session {s.id[:8]}", "HIGH",
                           user_id=actor.id, ip_address=ip, metadata={"session_id": s.id}, commit=False,
                           org_id=s.organization_id)
        alert_service.raise_alert(db, "FRAME_REPLAY_DETECTED", "HIGH", "user", actor.ref,
                                  "A previously submitted frame was submitted again", commit=False,
                                  org_id=s.organization_id)
        # A feed that keeps producing byte-identical frames is frozen or replayed: end the session.
        state = json.loads(s.state_json or "{}")
        state["_replays"] = int(state.get("_replays", 0)) + 1
        s.state_json = json.dumps(state)
        if state["_replays"] >= MAX_REPLAYED_FRAMES:
            _close(db, s, OutcomeEnum.LIVENESS_FAILED, "FRAME_REPLAY", now)
            db.commit()
            return session_view(db, s, actor, frame=None)
        db.commit()
        raise CaptureError(409, "FRAME_REPLAYED",
                           "The camera image is identical to an earlier frame — the feed may be frozen or replayed")

    # Atomic claim: single in-flight frame, ordered sequence, one-time token, frame budget.
    stale_before = now - timedelta(seconds=IN_FLIGHT_STALE_S)
    claimed = (db.query(VerificationSession)
               .filter(VerificationSession.id == s.id, VerificationSession.is_active == True,  # noqa: E712
                       VerificationSession.frame_seq == seq - 1,
                       VerificationSession.frame_token_hash == _hash(frame_token),
                       VerificationSession.frame_count < policy.max_frames_per_session,
                       or_(VerificationSession.in_flight == False,  # noqa: E712
                           VerificationSession.in_flight_since < stale_before))
               .update({"in_flight": True, "in_flight_since": now, "frame_seq": seq,
                        "frame_count": VerificationSession.frame_count + 1, "last_activity_at": now},
                       synchronize_session=False))
    db.commit()
    if claimed != 1:
        db.refresh(s)
        if s.in_flight:
            raise CaptureError(409, "FRAME_IN_FLIGHT", "A frame for this session is already being processed")
        if not s.is_active:
            raise CaptureError(409, "SESSION_CLOSED", "This session has already ended", payload=session_view(db, s, actor))
        raise CaptureError(409, "STALE_FRAME", "Frame token or sequence is not current")

    db.add(FrameDigest(digest=digest, session_id=s.id, created_at=now))
    db.commit()
    db.refresh(s)
    try:
        return _process_claimed(db, s, actor, policy, bgr, digest, now, ip)
    except CaptureError:
        raise
    except Exception:
        # Unexpected failure (e.g. database error): free the claim so the session is not wedged.
        db.rollback()
        try:
            db.query(VerificationSession).filter(VerificationSession.id == s.id).update(
                {"in_flight": False, "in_flight_since": None}, synchronize_session=False)
            db.commit()
        except Exception:  # noqa: BLE001
            db.rollback()
        raise


def _process_claimed(db: Session, s: VerificationSession, actor: Actor, policy: Policy, bgr: np.ndarray,
                     digest: str, now: datetime, ip: str) -> dict[str, Any]:
    samples: list[np.ndarray] = []
    if s.samples_encrypted:
        samples = list(decrypt_vectors(s.samples_encrypted).vectors)
    state = EngineState.from_json(s.state_json, samples)
    evaluated_step = state.step          # the step this frame is judged against
    engine = Engine(policy, s.purpose)
    reference_blob = _reference_blob(db, s)

    try:
        analysis, decision, timings = runtime.run(_run_engine, runtime.provider, bgr, engine, state, now,
                                                  reference_blob, timeout=30.0)
        errors = 0
    except InferenceBusy:
        _release_claim(db, s, digest)
        raise CaptureError(503, "INFERENCE_BUSY", "The server is busy processing frames. Retry shortly.",
                           headers={"Retry-After": "1"})
    except InferenceUnavailable as exc:
        _release_claim(db, s, digest)
        raise CaptureError(503, "MODELS_UNAVAILABLE", str(exc), headers={"Retry-After": "5"})
    except Exception:  # noqa: BLE001 — model/runtime failure; never becomes a match
        log.exception("Frame processing failed for session %s", s.id)
        state = EngineState.from_json(s.state_json, samples)   # discard partial mutations
        errors = _bump_errors(s)
        terminal = errors >= MAX_CONSECUTIVE_ERRORS
        decision = FrameDecision(OutcomeEnum.ERROR, terminal=terminal, reason="PROCESSING_ERROR",
                                 message="The frame could not be processed" + (" repeatedly" if terminal else ""),
                                 guidance=["Hold still"])
        analysis, timings = FrameAnalysis(width=0, height=0, faces=[]), {}

    processing_ms = (datetime.utcnow() - now).total_seconds() * 1000
    return _apply_decision(db, s, actor, engine, state, analysis, decision, timings, processing_ms, now, errors, ip,
                          evaluated_step)


def _reference_blob(db: Session, s: VerificationSession) -> Optional[bytes]:
    if s.purpose == SessionPurpose.VERIFY.value:
        enr = enrolment_service.get_enrolment(db, s.organization_id, s.target_user_id)
        if enr is None or not enr.is_usable:
            return None
        return enr.embedding_encrypted
    if s.purpose == SessionPurpose.DOCUMENT.value and s.document_check_id:
        doc = db.query(DocumentCheck).filter(DocumentCheck.id == s.document_check_id).first()
        return doc.portrait_template_encrypted if doc else None
    return None


def _bump_errors(s: VerificationSession) -> int:
    data = json.loads(s.state_json)
    n = int(data.get("_consecutive_errors", 0)) + 1
    return n


def _release_claim(db: Session, s: VerificationSession, digest: str) -> None:
    """Undo a claim when the frame was never processed (server busy), so the client can resend it."""
    db.query(VerificationSession).filter(VerificationSession.id == s.id).update(
        {"in_flight": False, "in_flight_since": None,
         "frame_seq": VerificationSession.frame_seq - 1, "frame_count": VerificationSession.frame_count - 1},
        synchronize_session=False)
    db.query(FrameDigest).filter(FrameDigest.session_id == s.id, FrameDigest.digest == digest).delete(
        synchronize_session=False)
    db.commit()


def _apply_decision(db: Session, s: VerificationSession, actor: Actor, engine: Engine, state: EngineState,
                    analysis: FrameAnalysis, decision: FrameDecision, timings: dict[str, float],
                    processing_ms: float, now: datetime, errors: int, ip: str,
                    evaluated_step: Optional[str] = None) -> dict[str, Any]:
    policy = engine.p
    token, token_hash = _new_token()
    state_dict = json.loads(state.to_json())
    state_dict["_consecutive_errors"] = errors
    model = getattr(runtime.provider, "recognition_model", "ArcFace")

    finalize_info: dict[str, Any] = {}
    outcome = decision.outcome
    reason = decision.reason
    message = decision.message
    enrol_samples = list(state.samples) if (decision.terminal and outcome == OutcomeEnum.ENROLMENT_CAPTURED) else []

    values: dict[str, Any] = {
        "in_flight": False, "in_flight_since": None, "state_json": json.dumps(state_dict),
        "attempts_used": state.attempts_used, "challenge_round": state.round, "challenge_index": state.index,
        "step_started_at": datetime.fromisoformat(state.step_started_at) if state.step_started_at else None,
        "samples_encrypted": encrypt_vectors(state.samples, model) if (state.samples and not decision.terminal) else None,
        "total_processing_ms": VerificationSession.total_processing_ms + processing_ms,
        "frame_token_hash": None if decision.terminal else token_hash,
    }
    if decision.terminal:
        values.update({"is_active": False, "final_outcome": outcome.value, "reason_code": reason,
                       "completed_at": now, "final_distance": decision.distance,
                       "threshold": policy.match_threshold if engine.reference_needed() else None})
    applied = (db.query(VerificationSession)
               .filter(VerificationSession.id == s.id, VerificationSession.is_active == True)  # noqa: E712
               .update(values, synchronize_session=False))
    if applied != 1:   # cancelled or expired while the frame was being processed
        db.rollback()
        db.refresh(s)
        return session_view(db, s, actor, frame=None)
    db.commit()
    state.samples = []

    if enrol_samples:
        target = db.query(User).filter(User.id == s.target_user_id).first()
        try:
            result = enrolment_service.finalize_capture(db, actor, target, s.id, enrol_samples, state.quality_scores,
                                                        policy, model, ip)
            finalize_info = {"request_id": result.request.id, "request_status": result.request.status,
                             "activated": result.activated, "approval_reasons": result.reasons}
            message = ("Enrolment complete — template active" if result.activated
                       else "Enrolment captured — awaiting administrator approval")
        except enrolment_service.EnrolmentQualityError as exc:
            outcome, reason, message = OutcomeEnum.ERROR, exc.code, exc.message
            db.query(VerificationSession).filter(VerificationSession.id == s.id).update(
                {"final_outcome": outcome.value, "reason_code": reason}, synchronize_session=False)
        enrol_samples = []

    face = decision.face
    db.add(VerificationEvent(
        organization_id=s.organization_id,
        session_id=s.id, user_id=s.target_user_id, timestamp=now, outcome=outcome.value, reason_code=reason,
        frame_seq=s.frame_seq, step_type=evaluated_step,
        distance=decision.distance, threshold=policy.match_threshold if decision.distance is not None else None,
        passive_label=decision.passive.label if decision.passive else None,
        passive_confidence=round(decision.passive.confidence, 4) if decision.passive else None,
        face_count=len(analysis.faces), quality_score=decision.quality.score if decision.quality else None,
        yaw=round(face.yaw, 2) if face else None, pitch=round(face.pitch, 2) if face else None,
        inference_ms=round(processing_ms, 1), explanation=message[:500], policy_version=str(policy.version),
    ))
    _apply_step_events(db, s, decision.step_events, now)
    db.commit()
    db.refresh(s)

    if decision.terminal:
        _on_terminal(db, s, policy, decision)
        if s.purpose == SessionPurpose.DOCUMENT.value:
            from backend.services import document_service
            document_service.complete_from_session(db, s)
        db.commit()
        db.refresh(s)

    frame = {
        "seq": s.frame_seq,
        "outcome": outcome.value,
        "reason": reason,
        "message": message,
        "guidance": decision.guidance,
        "terminal": decision.terminal,
        "new_round": decision.new_round,
        "face_count": len(analysis.faces),
        "face_box": list(decision.face.bbox) if decision.face else None,
        "pose": {"yaw": round(decision.face.yaw, 1), "pitch": round(decision.face.pitch, 1)} if decision.face else None,
        "quality": ({"score": decision.quality.score, "issues": decision.quality.issues,
                     "components": decision.quality.components} if decision.quality else None),
        "processing_ms": round(processing_ms, 1),
        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        **finalize_info,
    }
    if actor.can(Permission.HISTORY_SCORES):
        frame["distance"] = decision.distance
        frame["passive"] = ({"label": decision.passive.label, "confidence": round(decision.passive.confidence, 4)}
                            if decision.passive else None)
    view = session_view(db, s, actor, frame=frame, state=state)
    for hook in FRAME_HOOKS:
        try:
            hook(s, frame, view.get("challenge"))
        except Exception:  # noqa: BLE001 — progress fan-out must never affect the frame result
            log.exception("Frame hook failed for session %s", s.id)
    if not decision.terminal:
        view["frame_token"] = token
    return view


def _apply_step_events(db: Session, s: VerificationSession, events: list[StepEvent], now: datetime) -> None:
    for ev in events:
        row = (db.query(LivenessChallenge)
               .filter(LivenessChallenge.session_id == s.id, LivenessChallenge.round == ev.round,
                       LivenessChallenge.step_index == ev.index).first())
        if ev.status == "STARTED":
            if row is None:
                db.add(LivenessChallenge(session_id=s.id, round=ev.round, step_index=ev.index, step_type=ev.step_type,
                                         status="PENDING", started_at=now))
                db.flush()
            continue
        if row is None:
            row = LivenessChallenge(session_id=s.id, round=ev.round, step_index=ev.index, step_type=ev.step_type,
                                    started_at=now)
            db.add(row)
        row.status = ev.status
        row.completed_at = now
        row.evidence_json = json.dumps(ev.evidence) if ev.evidence else None
        db.flush()


def _close(db: Session, s: VerificationSession, outcome: OutcomeEnum, reason: str, now: datetime) -> None:
    s.is_active = False
    s.in_flight = False
    s.final_outcome = outcome.value
    s.reason_code = reason
    s.completed_at = now
    s.frame_token_hash = None
    s.samples_encrypted = None
    for ch in db.query(LivenessChallenge).filter(LivenessChallenge.session_id == s.id,
                                                  LivenessChallenge.status == "PENDING").all():
        ch.status = "CANCELLED" if outcome == OutcomeEnum.SESSION_CANCELLED else "EXPIRED"
        ch.completed_at = now
    db.add(VerificationEvent(organization_id=s.organization_id, session_id=s.id, user_id=s.target_user_id,
                             timestamp=now, outcome=outcome.value,
                             reason_code=reason, explanation=reason.replace("_", " ").capitalize(),
                             policy_version=str(s.policy_version)))
    if s.purpose == SessionPurpose.DOCUMENT.value:
        from backend.services import document_service
        document_service.complete_from_session(db, s)
    run_terminal_hooks(db, s, None)
    bus.publish("sessions", "session_closed", {"id": s.id, "outcome": outcome.value, "purpose": s.purpose,
                                               "channel": s.channel}, org_id=s.organization_id)


# Services that react to a session ending (cross-device pairing, manual review, integrations, cameras)
# register here, so this module does not import them.
TerminalHook = Callable[[Session, VerificationSession, Optional[FrameDecision]], None]
TERMINAL_HOOKS: list[TerminalHook] = []
# Per-frame observers (e.g. live progress for a paired desktop): fn(session, frame_view, challenge_view)
FRAME_HOOKS: list[Callable[[VerificationSession, dict, Optional[dict]], None]] = []


def run_terminal_hooks(db: Session, s: VerificationSession, decision: Optional[FrameDecision]) -> None:
    for hook in TERMINAL_HOOKS:
        try:
            with db.begin_nested():
                hook(db, s, decision)
        except Exception:  # noqa: BLE001 — a failing side effect must never change the decision
            log.exception("Terminal hook %s failed for session %s", getattr(hook, "__name__", hook), s.id)


def _on_terminal(db: Session, s: VerificationSession, policy: Policy, decision: Optional[FrameDecision] = None) -> None:
    if s.final_outcome == OutcomeEnum.LIVENESS_FAILED.value:
        log_security_event(db, "LIVENESS_FAILED", f"Liveness failed on session {s.id[:8]} ({s.reason_code})", "HIGH",
                           user_id=s.created_by_user_id, metadata={"session_id": s.id, "target_user_id": s.target_user_id},
                           commit=False, org_id=s.organization_id)
    alert_service.evaluate_terminal_session(db, s, policy)
    if s.camera_id:
        db.query(Camera).filter(Camera.id == s.camera_id).update({"last_verification_at": s.completed_at},
                                                                 synchronize_session=False)
    run_terminal_hooks(db, s, decision)
    bus.publish("sessions", "session_closed", {"id": s.id, "outcome": s.final_outcome, "purpose": s.purpose,
                                               "channel": s.channel, "review_status": s.review_status},
                org_id=s.organization_id)


def resync_session(db: Session, actor: Actor, session_id: str, ip: str) -> dict[str, Any]:
    """Re-issue a frame token after a lost response (network recovery). Owner only, never mid-frame."""
    s = load_owned_session(db, actor, session_id, ip)
    now = datetime.utcnow()
    if expire_if_needed(db, s, now):
        db.commit()
        return session_view(db, s, actor)
    if not s.is_active:
        return session_view(db, s, actor)
    token, token_hash = _new_token()
    stale_before = now - timedelta(seconds=IN_FLIGHT_STALE_S)
    n = (db.query(VerificationSession)
         .filter(VerificationSession.id == s.id, VerificationSession.is_active == True,  # noqa: E712
                 or_(VerificationSession.in_flight == False, VerificationSession.in_flight_since < stale_before))  # noqa: E712
         .update({"frame_token_hash": token_hash, "in_flight": False, "in_flight_since": None},
                 synchronize_session=False))
    db.commit()
    if n != 1:
        raise CaptureError(409, "FRAME_IN_FLIGHT", "A frame for this session is still being processed")
    log_security_event(db, "SESSION_RESYNC", f"Frame token re-issued for session {s.id[:8]}", "INFO",
                       user_id=actor.id, ip_address=ip, metadata={"session_id": s.id}, org_id=s.organization_id)
    db.refresh(s)
    view = session_view(db, s, actor)
    view["frame_token"] = token
    return view


def cancel_session(db: Session, actor: Actor, session_id: str, ip: str) -> dict[str, Any]:
    s = load_owned_session(db, actor, session_id, ip)
    if s.is_active:
        _close(db, s, OutcomeEnum.SESSION_CANCELLED, "USER_CANCELLED", datetime.utcnow())
        db.commit()
    return session_view(db, s, actor)


def expire_stale_sessions(db: Session) -> int:
    now = datetime.utcnow()
    n = 0
    for s in db.query(VerificationSession).filter(VerificationSession.is_active == True).all():  # noqa: E712
        if expire_if_needed(db, s, now):
            n += 1
    if n:
        db.commit()
    return n


def session_view(db: Session, s: VerificationSession, actor: Actor, frame: Optional[dict] = None,
                 state: Optional[EngineState] = None) -> dict[str, Any]:
    policy = _policy_of(s)
    if state is None and s.state_json:
        state = EngineState.from_json(s.state_json)
    view: dict[str, Any] = {
        "session_id": s.id,
        "purpose": s.purpose,
        "channel": s.channel,
        "camera_id": s.camera_id,
        "review_status": s.review_status,
        "target_user_id": s.target_user_id,
        "document_check_id": s.document_check_id,
        "status": "ACTIVE" if s.is_active else "CLOSED",
        "final_outcome": s.final_outcome,
        "reason_code": s.reason_code,
        "created_at": s.created_at,
        "expires_at": s.expires_at,
        "completed_at": s.completed_at,
        "frame_seq": s.frame_seq,
        "frames_processed": s.frame_count,
        "frames_remaining": max(0, policy.max_frames_per_session - s.frame_count),
        "attempts_used": s.attempts_used,
        "attempts_remaining": max(0, policy.max_attempts - s.attempts_used),
        "policy_version": s.policy_version,
        "capture": policy.capture_config(),
        "challenge": step_view(state) if (state and s.is_active) else None,
        "frame": frame,
    }
    if not s.is_active and actor.can(Permission.HISTORY_SCORES):
        view["final_distance"] = s.final_distance
        view["threshold"] = s.threshold
    return view
