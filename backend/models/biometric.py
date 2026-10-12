"""BiometricEnrolment and ConsentRecord models."""
from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Float, ForeignKey, LargeBinary, Text
from sqlalchemy.orm import relationship
from backend.database import Base


class BiometricEnrolment(Base):
    __tablename__ = "biometric_enrolments"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, unique=True)
    embedding_encrypted = Column(LargeBinary, nullable=True)
    consent_given = Column(Boolean, default=False, nullable=False)
    consent_timestamp = Column(DateTime, nullable=True)
    quality_score = Column(Float, default=0.0, nullable=False)
    enrolled_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    model_name = Column(String(50), default="ArcFace", nullable=False)

    # Relationships
    user = relationship("User", back_populates="enrolment")


class ConsentRecord(Base):
    __tablename__ = "consent_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    consent_given = Column(Boolean, nullable=False)
    consent_text = Column(Text, nullable=True)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ip_address = Column(String(45), nullable=True)
