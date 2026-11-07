"""Manual review of verification sessions and the notification centre."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint

from backend.database import Base


class ReviewStatus:
    PENDING = "PENDING"
    IN_REVIEW = "IN_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    RETRY_REQUESTED = "RETRY_REQUESTED"
    EXPIRED = "EXPIRED"


OPEN_REVIEW_STATUSES = frozenset({ReviewStatus.PENDING, ReviewStatus.IN_REVIEW})


class VerificationReview(Base):
    """A case raised for a human decision.

    The reviewer's decision is an *administrative disposition*; it never
    changes the session's technical outcome, never turns a confirmed spoof
    into a pass, and never issues an authentication session.
    """

    __tablename__ = "verification_reviews"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    session_id = Column(String(36), ForeignKey("verification_sessions.id"), nullable=True, unique=True)
    document_check_id = Column(String(36), ForeignKey("document_checks.id"), nullable=True)
    status = Column(String(20), nullable=False, default=ReviewStatus.PENDING, index=True)
    risk_level = Column(String(8), nullable=False, default="MEDIUM")
    reasons_json = Column(Text, nullable=False)
    technical_outcome = Column(String(32), nullable=True)
    spoof_confirmed = Column(Boolean, nullable=False, default=False)
    subject_user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    assigned_to_user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    assigned_at = Column(DateTime, nullable=True)
    decided_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    decided_at = Column(DateTime, nullable=True)
    decision_note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    version = Column(Integer, nullable=False, default=1)       # optimistic concurrency guard


class ReviewEvent(Base):
    __tablename__ = "review_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    review_id = Column(String(36), ForeignKey("verification_reviews.id", ondelete="CASCADE"), nullable=False, index=True)
    actor_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(24), nullable=False)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Notification(Base):
    """An in-app notification for one user in one organisation. Never carries biometric data."""

    __tablename__ = "notifications"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    type = Column(String(40), nullable=False, index=True)
    severity = Column(String(8), nullable=False, default="INFO")
    title = Column(String(200), nullable=False)
    body = Column(Text, nullable=True)
    link = Column(String(300), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    read_at = Column(DateTime, nullable=True)
    email_status = Column(String(16), nullable=False, default="NOT_REQUESTED")   # NOT_REQUESTED/PENDING/SENT/FAILED/UNAVAILABLE
    email_attempts = Column(Integer, nullable=False, default=0)
    email_next_attempt_at = Column(DateTime, nullable=True, index=True)
    email_last_error = Column(String(300), nullable=True)
    email_sent_at = Column(DateTime, nullable=True)


class NotificationPreference(Base):
    __tablename__ = "notification_preferences"
    __table_args__ = (UniqueConstraint("user_id", "type", name="uq_notification_pref"),)

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    type = Column(String(40), nullable=False)
    in_app = Column(Boolean, nullable=False, default=True)
    email = Column(Boolean, nullable=False, default=False)
