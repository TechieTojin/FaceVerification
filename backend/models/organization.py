"""Organisations (tenants), memberships and invitations.

A user account is global; what a user may do is decided by their membership
role in the *active* organisation. Every tenant-scoped row carries an
`organization_id`, and every query on such a row filters by the caller's
active organisation (see `backend.core.tenancy`).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Enum as SAEnum, ForeignKey, String, UniqueConstraint

from backend.database import Base
from backend.models.user import RoleEnum


class OrgStatus:
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    slug = Column(String(64), nullable=False, unique=True, index=True)
    name = Column(String(200), nullable=False)
    status = Column(String(16), nullable=False, default=OrgStatus.ACTIVE)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)


class MembershipStatus:
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("organization_id", "user_id", name="uq_membership_org_user"),)

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(SAEnum(RoleEnum, native_enum=False, length=16), nullable=False, default=RoleEnum.OPERATOR)
    status = Column(String(16), nullable=False, default=MembershipStatus.ACTIVE)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    invited_by_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)


class InvitationStatus:
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


class Invitation(Base):
    """An invitation to join an organisation. Only a hash of the token is stored."""

    __tablename__ = "organization_invitations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    email = Column(String(255), nullable=False, index=True)
    role = Column(SAEnum(RoleEnum, native_enum=False, length=16), nullable=False, default=RoleEnum.OPERATOR)
    token_hash = Column(String(64), nullable=False, unique=True)
    status = Column(String(16), nullable=False, default=InvitationStatus.PENDING, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_by_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    accepted_at = Column(DateTime, nullable=True)
    accepted_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
