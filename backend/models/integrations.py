"""Integration API: clients, scoped credentials, webhooks, signed assertions, idempotency."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, LargeBinary, String, Text, UniqueConstraint

from backend.database import Base


class IntegrationClient(Base):
    __tablename__ = "integration_clients"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    name = Column(String(120), nullable=False)
    description = Column(Text, nullable=True)
    scopes = Column(String(300), nullable=False, default="")
    status = Column(String(12), nullable=False, default="ACTIVE")        # ACTIVE / REVOKED
    rate_limit_per_minute = Column(Integer, nullable=False, default=60)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    revoked_at = Column(DateTime, nullable=True)


class ApiCredential(Base):
    """`fvk_<key_id>_<secret>`. Only an HMAC of the secret is stored."""

    __tablename__ = "api_credentials"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String(36), ForeignKey("integration_clients.id", ondelete="CASCADE"), nullable=False, index=True)
    key_id = Column(String(24), nullable=False, unique=True)
    secret_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    expires_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)


class WebhookEndpoint(Base):
    __tablename__ = "webhook_endpoints"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    client_id = Column(String(36), ForeignKey("integration_clients.id", ondelete="CASCADE"), nullable=False, index=True)
    url = Column(String(500), nullable=False)
    secret_encrypted = Column(LargeBinary, nullable=False)
    events = Column(String(300), nullable=False, default="verification.completed")
    status = Column(String(12), nullable=False, default="ACTIVE")         # ACTIVE / DISABLED
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    secret_rotated_at = Column(DateTime, nullable=True)


class WebhookDelivery(Base):
    __tablename__ = "webhook_deliveries"
    __table_args__ = (UniqueConstraint("endpoint_id", "event_id", name="uq_webhook_delivery_event"),)

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    endpoint_id = Column(String(36), ForeignKey("webhook_endpoints.id", ondelete="CASCADE"), nullable=False, index=True)
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    event_id = Column(String(36), nullable=False)
    event_type = Column(String(48), nullable=False)
    payload_json = Column(Text, nullable=False)
    status = Column(String(12), nullable=False, default="PENDING", index=True)   # PENDING / DELIVERED / FAILED
    attempts = Column(Integer, nullable=False, default=0)
    next_attempt_at = Column(DateTime, nullable=True, index=True)
    last_status_code = Column(Integer, nullable=True)
    last_error = Column(String(300), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    delivered_at = Column(DateTime, nullable=True)


class SigningKey(Base):
    """ES256 keys for verification assertions and kiosk bundles. Private key Fernet-encrypted."""

    __tablename__ = "signing_keys"

    id = Column(String(36), primary_key=True)                              # the JWK `kid`
    alg = Column(String(10), nullable=False, default="ES256")
    public_pem = Column(Text, nullable=False)
    private_encrypted = Column(LargeBinary, nullable=False)
    status = Column(String(12), nullable=False, default="ACTIVE")          # ACTIVE / RETIRED
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    retired_at = Column(DateTime, nullable=True)


class VerificationAssertion(Base):
    __tablename__ = "verification_assertions"

    id = Column(String(36), primary_key=True)                              # the JWT `jti`
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    client_id = Column(String(36), ForeignKey("integration_clients.id"), nullable=True, index=True)
    session_id = Column(String(36), ForeignKey("verification_sessions.id"), nullable=False, index=True)
    subject = Column(String(128), nullable=False)
    outcome = Column(String(32), nullable=False)
    kid = Column(String(36), nullable=False)
    issued_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    consumed_at = Column(DateTime, nullable=True)


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (UniqueConstraint("client_id", "key", name="uq_idempotency_client_key"),)

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String(36), ForeignKey("integration_clients.id", ondelete="CASCADE"), nullable=False)
    key = Column(String(128), nullable=False)
    request_hash = Column(String(64), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
