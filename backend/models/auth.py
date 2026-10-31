"""Refresh tokens and login attempts."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String

from backend.database import Base


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    token_hash = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    family_id = Column(String(36), nullable=True, index=True)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    amr = Column(String(64), nullable=True)            # authentication methods of the original sign-in
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Boolean, default=False, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    replaced_by = Column(String(64), nullable=True)
    user_agent = Column(String(200), nullable=True)
    ip_address = Column(String(45), nullable=True)


class LoginAttempt(Base):
    __tablename__ = "login_attempts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    identifier = Column(String(255), nullable=False, index=True)
    identifier_type = Column(String(10), nullable=False)
    success = Column(Boolean, nullable=False)
    attempted_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    ip_address = Column(String(45), nullable=True)
