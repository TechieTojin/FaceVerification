"""Passkeys (WebAuthn credentials), authentication challenges, MFA factors and recovery codes.

Secrets are never stored in clear: TOTP seeds are Fernet-encrypted, OTPs,
recovery codes and pending-login tokens are stored as keyed hashes.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, LargeBinary, String, Text

from backend.database import Base


class WebAuthnCredential(Base):
    __tablename__ = "webauthn_credentials"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    credential_id = Column(String(512), nullable=False, unique=True)          # base64url
    public_key = Column(LargeBinary, nullable=False)                          # COSE public key
    sign_count = Column(Integer, nullable=False, default=0)
    transports = Column(String(200), nullable=True)
    aaguid = Column(String(36), nullable=True)
    device_type = Column(String(24), nullable=True)                           # single_device / multi_device
    backed_up = Column(Boolean, nullable=False, default=False)
    name = Column(String(100), nullable=False, default="Passkey")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)


class ChallengeKind:
    WEBAUTHN_REGISTER = "WEBAUTHN_REGISTER"
    WEBAUTHN_LOGIN = "WEBAUTHN_LOGIN"
    WEBAUTHN_STEP_UP = "WEBAUTHN_STEP_UP"
    MFA_PENDING = "MFA_PENDING"          # password accepted, second factor outstanding
    FACE_PENDING = "FACE_PENDING"        # passkey accepted, face check outstanding
    EMAIL_OTP = "EMAIL_OTP"


class AuthChallenge(Base):
    """Server-side state for one authentication ceremony. Single use, short-lived."""

    __tablename__ = "auth_challenges"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    kind = Column(String(24), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    challenge = Column(String(200), nullable=True)                           # base64url WebAuthn challenge
    token_hash = Column(String(64), nullable=True, unique=True)             # pending-login token / OTP hash
    data_json = Column(Text, nullable=True)
    attempts = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    consumed_at = Column(DateTime, nullable=True)
    ip_address = Column(String(45), nullable=True)


class FactorKind:
    TOTP = "TOTP"
    EMAIL_OTP = "EMAIL_OTP"


class MfaFactor(Base):
    __tablename__ = "mfa_factors"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    kind = Column(String(16), nullable=False)
    label = Column(String(100), nullable=False, default="Authenticator app")
    secret_encrypted = Column(LargeBinary, nullable=True)
    status = Column(String(16), nullable=False, default="PENDING")           # PENDING / ACTIVE / REVOKED
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    confirmed_at = Column(DateTime, nullable=True)
    last_used_step = Column(Integer, nullable=True)                           # TOTP replay guard
    last_used_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)


class RecoveryCode(Base):
    __tablename__ = "recovery_codes"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    batch_id = Column(String(36), nullable=False, index=True)
    code_hash = Column(String(64), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    used_at = Column(DateTime, nullable=True)
