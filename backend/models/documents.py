"""Identity-document portrait checks, model metadata and evaluation runs."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text

from backend.database import Base


class DocumentCheck(Base):
    """Face-to-document-portrait comparison.

    The document image itself is never stored. Only an encrypted portrait
    template is kept, and only until the check completes or retention expires.
    Document *authenticity* is a separate field and is NOT_PERFORMED unless a
    configured provider actually ran.
    """

    __tablename__ = "document_checks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    created_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    target_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    subject_label = Column(String(200), nullable=True)
    status = Column(String(32), nullable=False, default="AWAITING_DOCUMENT", index=True)
    consent_given = Column(Boolean, nullable=False, default=False)
    consent_at = Column(DateTime, nullable=True)
    document_type = Column(String(24), nullable=False, default="ID_CARD")
    document_quality_score = Column(Float, nullable=True)
    document_quality_issues = Column(Text, nullable=True)
    portrait_detected = Column(Boolean, nullable=False, default=False)
    portrait_template_encrypted = Column(LargeBinary, nullable=True)
    face_match_result = Column(String(16), nullable=False, default="NOT_EVALUATED")
    face_match_distance = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    liveness_passed = Column(Boolean, nullable=True)
    authenticity_status = Column(String(16), nullable=False, default="NOT_PERFORMED")
    authenticity_provider = Column(String(32), nullable=True)
    authenticity_detail = Column(Text, nullable=True)
    overall_decision = Column(String(40), nullable=False, default="PENDING")
    session_id = Column(String(36), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)
    retention_expires_at = Column(DateTime, nullable=False)
    purged_at = Column(DateTime, nullable=True)


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(64), nullable=False, unique=True)
    task = Column(String(24), nullable=False)
    file_name = Column(String(128), nullable=True)
    sha256 = Column(String(64), nullable=True)
    embedding_dim = Column(Integer, nullable=True)
    approved = Column(Boolean, nullable=False, default=True)
    status = Column(String(16), nullable=False, default="NOT_LOADED")
    load_ms = Column(Float, nullable=True)
    error = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class EvaluationRun(Base):
    """Offline biometric evaluation on an administrator-supplied labelled dataset."""

    __tablename__ = "evaluation_runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    created_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(16), nullable=False, default="RUNNING")
    model_name = Column(String(50), nullable=False)
    dataset_label = Column(String(200), nullable=True)
    identity_count = Column(Integer, nullable=False, default=0)
    image_count = Column(Integer, nullable=False, default=0)
    skipped_images = Column(Integer, nullable=False, default=0)
    genuine_pairs = Column(Integer, nullable=False, default=0)
    impostor_pairs = Column(Integer, nullable=False, default=0)
    results_json = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
