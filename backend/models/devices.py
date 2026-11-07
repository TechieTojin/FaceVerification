"""Registered devices (managed kiosks / browser terminals), cameras and offline-kiosk records.

Trust model:
  * A device proves possession of a private key generated on the device during
    provisioning (ECDSA P-256). Only the public key is stored here.
  * KIOSK devices run the managed kiosk agent (`kiosk/`); BROWSER_TERMINAL
    devices are ordinary browsers holding a WebCrypto key. Neither is
    hardware-attested — the key's protection depends on the host.
  * Cameras are explicitly registered. The server never discovers cameras.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint

from backend.database import Base


class DeviceKind:
    KIOSK = "KIOSK"
    BROWSER_TERMINAL = "BROWSER_TERMINAL"


class DeviceStatus:
    PENDING = "PENDING"        # created, awaiting provisioning with the one-time code
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"


class Device(Base):
    __tablename__ = "devices"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    name = Column(String(120), nullable=False)
    kind = Column(String(20), nullable=False, default=DeviceKind.KIOSK)
    status = Column(String(12), nullable=False, default=DeviceStatus.PENDING, index=True)
    location = Column(String(200), nullable=True)
    public_key_pem = Column(Text, nullable=True)
    key_fingerprint = Column(String(64), nullable=True, index=True)
    key_protection = Column(String(32), nullable=True)            # as reported by the device, e.g. "DPAPI", "software"
    provisioning_code_hash = Column(String(64), nullable=True, unique=True)
    provisioning_expires_at = Column(DateTime, nullable=True)
    software_version = Column(String(40), nullable=True)
    platform = Column(String(80), nullable=True)
    health_json = Column(Text, nullable=True)
    last_seen_at = Column(DateTime, nullable=True)
    last_ip = Column(String(45), nullable=True)
    offline_enabled = Column(Boolean, nullable=False, default=False)
    offline_validity_hours = Column(Integer, nullable=False, default=24)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    activated_at = Column(DateTime, nullable=True)
    key_rotated_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    revoked_by_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    revoke_reason = Column(String(200), nullable=True)


class DeviceNonce(Base):
    """Nonces of signed device requests, kept for the replay window."""

    __tablename__ = "device_nonces"
    __table_args__ = (UniqueConstraint("device_id", "nonce", name="uq_device_nonce"),)

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    device_id = Column(String(36), ForeignKey("devices.id", ondelete="CASCADE"), nullable=False)
    nonce = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    device_id = Column(String(36), ForeignKey("devices.id", ondelete="SET NULL"), nullable=True, index=True)
    label = Column(String(120), nullable=False)
    location = Column(String(200), nullable=True)
    hardware_label = Column(String(200), nullable=True)          # label the browser/agent reports for the camera
    capabilities_json = Column(Text, nullable=True)
    status = Column(String(12), nullable=False, default="ACTIVE", index=True)   # ACTIVE / DISABLED
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    last_heartbeat_at = Column(DateTime, nullable=True)
    last_heartbeat_by = Column(String(64), nullable=True)
    last_verification_at = Column(DateTime, nullable=True)
    disabled_at = Column(DateTime, nullable=True)


class OfflineSubject(Base):
    """A subject explicitly authorised for offline verification on one kiosk."""

    __tablename__ = "offline_subjects"
    __table_args__ = (UniqueConstraint("device_id", "user_id", name="uq_offline_subject"),)

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    device_id = Column(String(36), ForeignKey("devices.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    added_by_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class OfflineBundle(Base):
    """A provisioning bundle issued to a kiosk (the encrypted payload itself is not retained)."""

    __tablename__ = "offline_bundles"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    device_id = Column(String(36), ForeignKey("devices.id", ondelete="CASCADE"), nullable=False, index=True)
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    issued_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    valid_until = Column(DateTime, nullable=False)
    subject_count = Column(Integer, nullable=False, default=0)
    policy_version = Column(Integer, nullable=True)
    payload_sha256 = Column(String(64), nullable=False)
    status = Column(String(12), nullable=False, default="ISSUED")        # ISSUED / SUPERSEDED / REVOKED


class OfflineEvent(Base):
    """An audit event recorded by a kiosk while offline and uploaded later. (device_id, event_id) is unique."""

    __tablename__ = "offline_events"
    __table_args__ = (UniqueConstraint("device_id", "event_id", name="uq_offline_event"),)

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    device_id = Column(String(36), ForeignKey("devices.id", ondelete="CASCADE"), nullable=False, index=True)
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    event_id = Column(String(36), nullable=False)
    seq = Column(Integer, nullable=False)
    event_type = Column(String(32), nullable=False)
    occurred_at = Column(DateTime, nullable=False)
    received_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    bundle_id = Column(String(36), nullable=True)
    subject_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    outcome = Column(String(32), nullable=True)
    reason_code = Column(String(48), nullable=True)
    payload_json = Column(Text, nullable=True)
    prev_hash = Column(String(64), nullable=True)
    chain_hash = Column(String(64), nullable=False)
    integrity = Column(String(12), nullable=False, default="OK")          # OK / BROKEN_CHAIN / LATE (outside validity)
