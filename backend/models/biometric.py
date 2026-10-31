"""Biometric enrolment, consent history and enrolment approval requests."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import relationship

from backend.database import Base


class EnrolmentStatus:
    NOT_ENROLLED = "NOT_ENROLLED"
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    WITHDRAWN = "WITHDRAWN"
    REENROLMENT_REQUIRED = "REENROLMENT_REQUIRED"


class BiometricEnrolment(Base):
    """One row per (organisation, user). Holds the *active* encrypted reference template.

    A template collected under one organisation's consent is never usable by another.
    """

    __tablename__ = "biometric_enrolments"
    __table_args__ = (UniqueConstraint("organization_id", "user_id", name="uq_enrolment_org_user"),)

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    status = Column(String(24), nullable=False, default=EnrolmentStatus.NOT_ENROLLED)
    embedding_encrypted = Column(LargeBinary, nullable=True)
    template_version = Column(Integer, nullable=False, default=2)
    embedding_dim = Column(Integer, nullable=True)
    model_name = Column(String(50), default="ArcFace", nullable=False)
    sample_count = Column(Integer, nullable=False, default=0)
    quality_score = Column(Float, default=0.0, nullable=False)
    consistency_score = Column(Float, nullable=True)
    consent_given = Column(Boolean, default=False, nullable=False)
    consent_timestamp = Column(DateTime, nullable=True)
    enrolled_at = Column(DateTime, nullable=True)
    enrolled_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    approved_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    revoked_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    reenrolment_requested = Column(Boolean, nullable=False, default=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", foreign_keys=[user_id])

    @property
    def is_usable(self) -> bool:
        return (
            self.status == EnrolmentStatus.ACTIVE
            and self.consent_given
            and self.embedding_encrypted is not None
            and self.template_version >= 2
        )


class ConsentRecord(Base):
    """Append-only consent history (grants and withdrawals)."""

    __tablename__ = "consent_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    consent_given = Column(Boolean, nullable=False)
    consent_text = Column(Text, nullable=True)
    consent_version = Column(String(20), nullable=True)
    purpose = Column(String(40), nullable=False, default="BIOMETRIC_ENROLMENT")
    method = Column(String(30), nullable=True)
    actor_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ip_address = Column(String(45), nullable=True)


class RequestStatus:
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    AUTO_APPROVED = "AUTO_APPROVED"
    CANCELLED = "CANCELLED"


class EnrolmentRequest(Base):
    """Every enrolment attempt that produced a candidate template.

    Sensitive cases (replacing an existing template, privileged target,
    suspected duplicate) stay PENDING until an administrator approves.
    """

    __tablename__ = "enrolment_requests"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    requested_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    session_id = Column(String(36), nullable=True)
    kind = Column(String(16), nullable=False, default="INITIAL")
    status = Column(String(16), nullable=False, default=RequestStatus.PENDING, index=True)
    approval_reasons = Column(Text, nullable=True)
    candidate_template_encrypted = Column(LargeBinary, nullable=True)
    embedding_dim = Column(Integer, nullable=True)
    model_name = Column(String(50), nullable=False, default="ArcFace")
    sample_count = Column(Integer, nullable=False, default=0)
    quality_score = Column(Float, nullable=False, default=0.0)
    consistency_score = Column(Float, nullable=True)
    duplicate_user_ids = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reviewed_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    review_note = Column(Text, nullable=True)
