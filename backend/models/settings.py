"""Versioned verification policy and append-only audit log."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint

from backend.database import Base


class PolicyVersion(Base):
    """Each change to the verification policy creates a new immutable version row."""

    __tablename__ = "policy_versions"
    __table_args__ = (UniqueConstraint("organization_id", "version", name="uq_policy_org_version"),)

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    version = Column(Integer, nullable=False, index=True)
    values_json = Column(Text, nullable=False)
    change_summary = Column(Text, nullable=True)
    is_sensitive_change = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)


class AuditLog(Base):
    """Who changed what. Never stores biometric data or secrets."""

    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    actor_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    action = Column(String(64), nullable=False, index=True)
    target_type = Column(String(32), nullable=True)
    target_id = Column(String(64), nullable=True, index=True)
    details_json = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
