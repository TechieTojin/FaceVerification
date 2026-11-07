"""Cross-device capture links: desktop→phone QR pairings and hosted integration links.

The QR / link carries only an opaque one-time token. Its hash is stored here.
Claiming it issues a separate pairing secret to the claiming browser; that
secret (hash stored) is what authorises the phone's capture requests.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text

from backend.database import Base


class PairingKind:
    QR = "QR"            # operator desktop shows QR, subject's phone captures
    HOSTED = "HOSTED"    # integration API link, subject opens it on any device


class PairingStatus:
    WAITING = "WAITING"                    # QR displayed, not yet scanned
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"   # phone claimed, operator must confirm the code
    READY = "READY"                        # confirmed; phone may start capture
    IN_PROGRESS = "IN_PROGRESS"            # capture session running
    COMPLETED = "COMPLETED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


OPEN_PAIRING_STATUSES = frozenset({PairingStatus.WAITING, PairingStatus.AWAITING_CONFIRMATION,
                                   PairingStatus.READY, PairingStatus.IN_PROGRESS})


class CapturePairing(Base):
    __tablename__ = "capture_pairings"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    kind = Column(String(8), nullable=False, default=PairingKind.QR)
    status = Column(String(24), nullable=False, default=PairingStatus.WAITING, index=True)
    token_hash = Column(String(64), nullable=False, unique=True)
    secret_hash = Column(String(64), nullable=True, unique=True)
    confirm_code_hash = Column(String(64), nullable=True)
    confirm_attempts = Column(Integer, nullable=False, default=0)
    target_user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    created_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    integration_client_id = Column(String(36), ForeignKey("integration_clients.id"), nullable=True, index=True)
    external_ref = Column(String(128), nullable=True)
    camera_id = Column(String(36), ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    claimed_at = Column(DateTime, nullable=True)
    claim_ip = Column(String(45), nullable=True)
    claim_user_agent = Column(String(200), nullable=True)
    confirmed_at = Column(DateTime, nullable=True)
    session_id = Column(String(36), nullable=True, index=True)
    sessions_started = Column(Integer, nullable=False, default=0)
    result_outcome = Column(String(32), nullable=True)
    completed_at = Column(DateTime, nullable=True)
    cancelled_reason = Column(String(48), nullable=True)
    metadata_json = Column(Text, nullable=True)
