"""User account model and role enum.

Accounts are global. Roles are per organisation (see `Membership`).
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, String
from sqlalchemy.orm import relationship

from backend.database import Base


class RoleEnum(str, enum.Enum):
    ADMIN = "ADMIN"
    OPERATOR = "OPERATOR"
    VIEWER = "VIEWER"


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_platform_admin = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login_at = Column(DateTime, nullable=True)
    password_changed_at = Column(DateTime, nullable=True)
    disabled_at = Column(DateTime, nullable=True)
    last_org_id = Column(String(36), nullable=True)

    security_events = relationship("SecurityEvent", back_populates="user")
