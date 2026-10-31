"""Capture sessions, per-frame events, liveness challenges, security events and alerts."""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text,
)
from sqlalchemy.orm import relationship

from backend.database import Base


class SessionPurpose(str, enum.Enum):
    VERIFY = "VERIFY"
    ENROL = "ENROL"
    DOCUMENT = "DOCUMENT"


class OutcomeEnum(str, enum.Enum):
    """Frame-level and session-level outcomes.

    Frame-level (non-terminal): IN_PROGRESS, RETRY_REQUIRED, NO_FACE, MULTIPLE_FACES, QUALITY_FAILED.
    Terminal: VERIFIED, NOT_MATCHED, LIVENESS_FAILED, SESSION_EXPIRED, SESSION_CANCELLED,
              ATTEMPTS_EXHAUSTED, ERROR, ENROLMENT_CAPTURED.
    """

    IN_PROGRESS = "IN_PROGRESS"
    RETRY_REQUIRED = "RETRY_REQUIRED"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    QUALITY_FAILED = "QUALITY_FAILED"
    VERIFIED = "VERIFIED"
    NOT_MATCHED = "NOT_MATCHED"
    LIVENESS_FAILED = "LIVENESS_FAILED"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    SESSION_CANCELLED = "SESSION_CANCELLED"
    ATTEMPTS_EXHAUSTED = "ATTEMPTS_EXHAUSTED"
    ENROLMENT_CAPTURED = "ENROLMENT_CAPTURED"
    ERROR = "ERROR"


TERMINAL_OUTCOMES = frozenset({
    OutcomeEnum.VERIFIED, OutcomeEnum.NOT_MATCHED, OutcomeEnum.LIVENESS_FAILED,
    OutcomeEnum.SESSION_EXPIRED, OutcomeEnum.SESSION_CANCELLED, OutcomeEnum.ATTEMPTS_EXHAUSTED,
    OutcomeEnum.ENROLMENT_CAPTURED, OutcomeEnum.ERROR,
})


class VerificationSession(Base):
    __tablename__ = "verification_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    # Who may drive this session: USER (operator), PAIRING (paired phone / hosted link), LOGIN (passkey login face check).
    owner_kind = Column(String(16), nullable=False, default="USER")
    owner_ref = Column(String(36), nullable=False, index=True)
    channel = Column(String(16), nullable=False, default="DESKTOP")
    camera_id = Column(String(36), ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True)
    pairing_id = Column(String(36), nullable=True, index=True)
    review_status = Column(String(20), nullable=True, index=True)
    risk_json = Column(Text, nullable=True)
    purpose = Column(String(16), nullable=False, default=SessionPurpose.VERIFY.value, index=True)
    target_user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    created_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    document_check_id = Column(String(36), ForeignKey("document_checks.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    final_outcome = Column(String(32), nullable=True, index=True)
    reason_code = Column(String(48), nullable=True)
    completed_at = Column(DateTime, nullable=True)
    frame_count = Column(Integer, default=0, nullable=False)
    attempts_used = Column(Integer, default=0, nullable=False)
    frame_seq = Column(Integer, default=0, nullable=False)
    frame_token_hash = Column(String(64), nullable=True)
    in_flight = Column(Boolean, default=False, nullable=False)
    in_flight_since = Column(DateTime, nullable=True)
    last_activity_at = Column(DateTime, nullable=True)
    policy_version = Column(Integer, nullable=True)
    policy_snapshot = Column(Text, nullable=True)
    challenge_round = Column(Integer, default=0, nullable=False)
    challenge_index = Column(Integer, default=0, nullable=False)
    step_started_at = Column(DateTime, nullable=True)
    state_json = Column(Text, nullable=True)
    samples_encrypted = Column(LargeBinary, nullable=True)
    final_distance = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    total_processing_ms = Column(Float, default=0.0, nullable=False)

    events = relationship("VerificationEvent", back_populates="session", cascade="all, delete-orphan")
    challenges = relationship(
        "LivenessChallenge", back_populates="session", cascade="all, delete-orphan",
        order_by="(LivenessChallenge.round, LivenessChallenge.step_index)",
    )


class VerificationEvent(Base):
    """One row per processed frame (never contains image data)."""

    __tablename__ = "verification_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    session_id = Column(String(36), ForeignKey("verification_sessions.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    outcome = Column(String(32), nullable=False, index=True)
    reason_code = Column(String(48), nullable=True)
    frame_seq = Column(Integer, nullable=True)
    step_type = Column(String(24), nullable=True)
    distance = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    passive_label = Column(String(8), nullable=True)
    passive_confidence = Column(Float, nullable=True)
    face_count = Column(Integer, default=0, nullable=False)
    quality_score = Column(Float, nullable=True)
    yaw = Column(Float, nullable=True)
    pitch = Column(Float, nullable=True)
    inference_ms = Column(Float, nullable=True)
    explanation = Column(Text, nullable=True)
    policy_version = Column(String(50), nullable=True)

    session = relationship("VerificationSession", back_populates="events")


class LivenessChallenge(Base):
    """A server-generated active-liveness step and the evidence that satisfied it."""

    __tablename__ = "liveness_challenges"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("verification_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    round = Column(Integer, nullable=False, default=0)
    step_index = Column(Integer, nullable=False)
    step_type = Column(String(24), nullable=False)
    status = Column(String(16), nullable=False, default="PENDING")
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    evidence_json = Column(Text, nullable=True)

    session = relationship("VerificationSession", back_populates="challenges")


class FrameDigest(Base):
    """SHA-256 of recently submitted frames — exact-replay detection across sessions."""

    __tablename__ = "frame_digests"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    digest = Column(String(64), nullable=False, index=True)
    session_id = Column(String(36), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class SecurityEvent(Base):
    __tablename__ = "security_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    description = Column(Text, nullable=True)
    severity = Column(String(20), default="INFO", nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    ip_address = Column(String(45), nullable=True)
    metadata_json = Column(Text, nullable=True)

    user = relationship("User", back_populates="security_events")


class SecurityAlert(Base):
    __tablename__ = "security_alerts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=True, index=True)
    rule = Column(String(48), nullable=False, index=True)
    severity = Column(String(12), nullable=False, default="MEDIUM")
    status = Column(String(16), nullable=False, default="OPEN", index=True)
    subject_type = Column(String(16), nullable=True)
    subject_id = Column(String(64), nullable=True, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    occurrence_count = Column(Integer, nullable=False, default=1)
    first_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    acknowledged_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    resolution_note = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=True)
