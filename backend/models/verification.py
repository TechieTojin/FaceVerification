"""VerificationSession, VerificationEvent, SecurityEvent models."""
from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Float, ForeignKey, Integer, Text, Enum as SAEnum
from sqlalchemy.orm import relationship
import enum
from backend.database import Base


class OutcomeEnum(str, enum.Enum):
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    SPOOF_DETECTED = "SPOOF_DETECTED"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    LOCKOUT = "LOCKOUT"


class VerificationSession(Base):
    __tablename__ = "verification_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    target_user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    created_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    final_outcome = Column(SAEnum(OutcomeEnum), nullable=True)
    completed_at = Column(DateTime, nullable=True)
    frame_count = Column(Integer, default=0, nullable=False)

    # Relationships
    target_user = relationship("User", foreign_keys=[target_user_id], back_populates="verification_sessions")
    events = relationship("VerificationEvent", back_populates="session")


class VerificationEvent(Base):
    __tablename__ = "verification_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("verification_sessions.id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    outcome = Column(SAEnum(OutcomeEnum), nullable=False)
    distance = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    liveness_score = Column(Float, nullable=True)
    face_count = Column(Integer, default=0, nullable=False)
    inference_ms = Column(Float, nullable=True)
    explanation = Column(Text, nullable=True)
    policy_version = Column(String(50), nullable=True)
    is_liveness_real = Column(Boolean, nullable=True)

    # Relationships
    session = relationship("VerificationSession", back_populates="events")


class SecurityEvent(Base):
    __tablename__ = "security_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    event_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    severity = Column(String(20), default="INFO", nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    ip_address = Column(String(45), nullable=True)
    metadata_json = Column(Text, nullable=True)

    # Relationships
    user = relationship("User", back_populates="security_events")
