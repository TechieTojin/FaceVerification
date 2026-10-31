"""Registered devices (managed kiosks, browser terminals), cameras, and offline-kiosk provisioning/sync.

Device trust:
  * An administrator creates a device and receives a one-time provisioning code (15 min).
  * The device generates an ECDSA P-256 key pair locally and sends only the public key with
    the code. From then on every device request is signed:
        X-Device-Id, X-Device-Timestamp, X-Device-Nonce, X-Device-Signature
        signature = ECDSA-SHA256 over "METHOD\\nPATH\\nTIMESTAMP\\nNONCE\\nSHA256(body)"
    Timestamps must be within ±5 minutes and each nonce is accepted once (replay protection).
  * This proves possession of the provisioned key, not hardware identity. Whether the key is
    hardware-protected depends on the device; the reported `key_protection` is shown as reported.
  * Revocation takes effect on the next request; a disconnected kiosk learns of it when it
    reconnects, which is why offline validity is bounded.

Offline bundles: only subjects explicitly authorised for one kiosk, encrypted to that kiosk's
public key (ephemeral ECDH P-256 → HKDF-SHA256 → AES-256-GCM) and signed by the server (ES256).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.config import settings
from backend.core import crypto
from backend.core.deps import Actor, membership_role
from backend.core.errors import ApiError
from backend.core.events import bus
from backend.core.permissions import Permission
from backend.models.devices import (
    Camera, Device, DeviceKind, DeviceNonce, DeviceStatus, OfflineBundle, OfflineEvent, OfflineSubject,
)
from backend.models.user import User
from backend.services import alert_service, notification_service
from backend.services.audit_service import audit, log_security_event
from backend.services.policy_service import get_active_policy

PROVISIONING_TTL = timedelta(minutes=15)
CLOCK_SKEW_S = 300
ONLINE_WINDOW_S = 120
CAMERA_ONLINE_WINDOW_S = 60
BUNDLE_INFO = b"fv-offline-bundle-v1"


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# ── administration ──────────────────────────────────────────────────────────

def create_device(db: Session, actor: Actor, name: str, kind: str, location: str, offline_enabled: bool,
                  offline_validity_hours: int, ip: str) -> tuple[Device, str]:
    if kind not in (DeviceKind.KIOSK, DeviceKind.BROWSER_TERMINAL):
        raise ApiError(422, "INVALID_KIND", "Device kind must be KIOSK or BROWSER_TERMINAL")
    if offline_enabled and kind != DeviceKind.KIOSK:
        raise ApiError(422, "OFFLINE_KIOSK_ONLY", "Offline verification is only available on managed kiosks")
    policy = get_active_policy(db, actor.org_id)
    validity = min(int(offline_validity_hours), int(policy.offline_validity_hours), settings.offline_max_validity_hours)
    code = crypto.token(12).replace("-", "").replace("_", "")[:12].upper()
    d = Device(organization_id=actor.org_id, name=name.strip()[:120], kind=kind, status=DeviceStatus.PENDING,
               location=(location or "").strip()[:200] or None, offline_enabled=offline_enabled,
               offline_validity_hours=max(1, validity), created_by_user_id=actor.id,
               provisioning_code_hash=crypto.keyed_hash(code, "provision"),
               provisioning_expires_at=datetime.utcnow() + PROVISIONING_TTL)
    db.add(d)
    db.commit()
    audit(db, actor.id, "DEVICE_CREATED", "device", d.id, {"name": d.name, "kind": kind, "offline": offline_enabled},
          ip, org_id=actor.org_id)
    return d, code


def load_device(db: Session, actor: Actor, device_id: str) -> Device:
    d = db.query(Device).filter(Device.id == device_id, Device.organization_id == actor.org_id).first()
    if d is None:
        raise ApiError(404, "NOT_FOUND", "Device not found")
    return d


def new_provisioning_code(db: Session, actor: Actor, device_id: str, ip: str) -> tuple[Device, str]:
    d = load_device(db, actor, device_id)
    if d.status != DeviceStatus.PENDING:
        raise ApiError(409, "ALREADY_PROVISIONED", "This device is already provisioned or revoked")
    code = crypto.token(12).replace("-", "").replace("_", "")[:12].upper()
    d.provisioning_code_hash = crypto.keyed_hash(code, "provision")
    d.provisioning_expires_at = datetime.utcnow() + PROVISIONING_TTL
    db.commit()
    audit(db, actor.id, "DEVICE_CODE_REISSUED", "device", d.id, None, ip, org_id=actor.org_id)
    return d, code


def revoke(db: Session, actor: Actor, device_id: str, reason: str, ip: str) -> Device:
    d = load_device(db, actor, device_id)
    if d.status == DeviceStatus.REVOKED:
        return d
    now = datetime.utcnow()
    d.status, d.revoked_at, d.revoked_by_user_id, d.revoke_reason = DeviceStatus.REVOKED, now, actor.id, reason[:200] or None
    d.provisioning_code_hash = None
    db.query(OfflineBundle).filter(OfflineBundle.device_id == d.id, OfflineBundle.status == "ISSUED").update(
        {"status": "REVOKED"}, synchronize_session=False)
    db.commit()
    audit(db, actor.id, "DEVICE_REVOKED", "device", d.id, {"reason": reason[:200]}, ip, org_id=actor.org_id)
    log_security_event(db, "DEVICE_REVOKED", f"Device {d.name} revoked", "WARNING", user_id=actor.id, ip_address=ip,
                       org_id=actor.org_id, metadata={"device_id": d.id})
    notification_service.notify_permission(db, actor.org_id, "DEVICE_REVOKED", f"Device revoked: {d.name}",
                                           reason[:300] or "A device was revoked.", link=f"/devices/{d.id}",
                                           permission=Permission.DEVICES_MANAGE.value)
    db.commit()
    bus.publish("devices", "device_updated", {"id": d.id, "status": d.status}, org_id=actor.org_id)
    return d


def device_status(d: Device) -> str:
    if d.status != DeviceStatus.ACTIVE:
        return d.status
    if d.last_seen_at is None:
        return "NEVER_CONNECTED"
    return "ONLINE" if (datetime.utcnow() - d.last_seen_at).total_seconds() <= ONLINE_WINDOW_S else "OFFLINE"


def device_view(db: Session, d: Device, detail: bool = False) -> dict[str, Any]:
    out = {"id": d.id, "name": d.name, "kind": d.kind, "status": d.status, "connection": device_status(d),
           "location": d.location, "software_version": d.software_version, "platform": d.platform,
           "key_fingerprint": d.key_fingerprint, "key_protection": d.key_protection,
           "hardware_attested": False, "last_seen_at": d.last_seen_at, "created_at": d.created_at,
           "activated_at": d.activated_at, "revoked_at": d.revoked_at, "revoke_reason": d.revoke_reason,
           "offline_enabled": d.offline_enabled, "offline_validity_hours": d.offline_validity_hours,
           "health": json.loads(d.health_json) if d.health_json else None,
           "provisioning_expires_at": d.provisioning_expires_at if d.status == DeviceStatus.PENDING else None}
    if detail:
        out["cameras"] = [camera_view(c) for c in db.query(Camera).filter(Camera.device_id == d.id)]
        subs = (db.query(OfflineSubject, User).join(User, User.id == OfflineSubject.user_id)
                .filter(OfflineSubject.device_id == d.id).all())
        out["offline_subjects"] = [{"user_id": u.id, "full_name": u.full_name, "added_at": s.created_at}
                                   for s, u in subs]
        b = (db.query(OfflineBundle).filter(OfflineBundle.device_id == d.id)
             .order_by(OfflineBundle.issued_at.desc()).first())
        out["latest_bundle"] = ({"id": b.id, "issued_at": b.issued_at, "valid_until": b.valid_until,
                                 "subject_count": b.subject_count, "status": b.status} if b else None)
        out["offline_events"] = db.query(OfflineEvent).filter(OfflineEvent.device_id == d.id).count()
    return out


# ── provisioning and signed requests ────────────────────────────────────────

def _fingerprint(pub: ec.EllipticCurvePublicKey) -> str:
    der = pub.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return hashlib.sha256(der).hexdigest()


def _load_public_key(pem: str) -> ec.EllipticCurvePublicKey:
    try:
        pub = serialization.load_pem_public_key(pem.encode())
    except (ValueError, TypeError) as exc:
        raise ApiError(422, "INVALID_PUBLIC_KEY", "Public key must be a PEM-encoded EC P-256 key") from exc
    if not isinstance(pub, ec.EllipticCurvePublicKey) or not isinstance(pub.curve, ec.SECP256R1):
        raise ApiError(422, "INVALID_PUBLIC_KEY", "Public key must be EC P-256")
    return pub


def provision(db: Session, code: str, public_key_pem: str, software_version: str, platform: str,
              key_protection: str, ip: str) -> Device:
    pub = _load_public_key(public_key_pem)
    now = datetime.utcnow()
    d = (db.query(Device)
         .filter(Device.provisioning_code_hash == crypto.keyed_hash((code or "").strip().upper(), "provision")).first())
    if d is None or d.status != DeviceStatus.PENDING or (d.provisioning_expires_at and d.provisioning_expires_at < now):
        log_security_event(db, "DEVICE_PROVISIONING_FAILED", "Invalid, used or expired provisioning code", "WARNING",
                           ip_address=ip, org_id=d.organization_id if d else None)
        raise ApiError(401, "INVALID_PROVISIONING_CODE", "The provisioning code is invalid, used or expired")
    n = (db.query(Device)
         .filter(Device.id == d.id, Device.status == DeviceStatus.PENDING, Device.provisioning_code_hash.isnot(None))
         .update({"status": DeviceStatus.ACTIVE, "public_key_pem": public_key_pem, "key_fingerprint": _fingerprint(pub),
                  "key_protection": (key_protection or "unknown")[:32], "software_version": (software_version or "")[:40],
                  "platform": (platform or "")[:80], "provisioning_code_hash": None, "activated_at": now,
                  "last_seen_at": now, "last_ip": ip[:45]}, synchronize_session=False))
    db.commit()
    if n != 1:
        raise ApiError(401, "INVALID_PROVISIONING_CODE", "The provisioning code is invalid, used or expired")
    db.refresh(d)
    audit(db, None, "DEVICE_PROVISIONED", "device", d.id, {"fingerprint": d.key_fingerprint[:16],
                                                          "key_protection": d.key_protection}, ip,
          org_id=d.organization_id)
    notification_service.notify_permission(db, d.organization_id, "DEVICE_REGISTERED", f"New device registered: {d.name}",
                                           f"{d.kind.replace('_', ' ').title()} activated ({d.key_protection} key).",
                                           link=f"/devices/{d.id}", permission=Permission.DEVICES_MANAGE.value)
    db.commit()
    bus.publish("devices", "device_updated", {"id": d.id, "status": d.status}, org_id=d.organization_id)
    return d


def canonical(method: str, path: str, ts: str, nonce: str, body: bytes) -> bytes:
    return f"{method.upper()}\n{path}\n{ts}\n{nonce}\n{hashlib.sha256(body).hexdigest()}".encode()


def authenticate_request(db: Session, device_id: str, ts: str, nonce: str, signature: str, method: str, path: str,
                         body: bytes, ip: str) -> Device:
    bad = ApiError(401, "INVALID_DEVICE_SIGNATURE", "Device request signature is not valid")
    d = db.query(Device).filter(Device.id == (device_id or "")[:36]).first()
    if d is None:
        raise bad
    if d.status == DeviceStatus.REVOKED:
        log_security_event(db, "REVOKED_DEVICE_ACCESS", f"Revoked device {d.name} attempted access", "HIGH",
                           ip_address=ip, org_id=d.organization_id, metadata={"device_id": d.id})
        alert_service.raise_alert(db, "REVOKED_DEVICE_ACCESS", "HIGH", "device", d.id,
                                  f"Revoked device {d.name} attempted to connect from {ip}", org_id=d.organization_id)
        raise ApiError(403, "DEVICE_REVOKED", "This device has been revoked")
    if d.status != DeviceStatus.ACTIVE or not d.public_key_pem:
        raise bad
    try:
        t = int(ts)
    except (TypeError, ValueError):
        raise bad from None
    if abs(time.time() - t) > CLOCK_SKEW_S:
        raise ApiError(401, "CLOCK_SKEW", "Device clock is too far from server time")
    if not nonce or len(nonce) < 16 or len(nonce) > 64:
        raise bad
    try:
        serialization.load_pem_public_key(d.public_key_pem.encode()).verify(
            _unb64(signature or ""), canonical(method, path, ts, nonce, body), ec.ECDSA(hashes.SHA256()))
    except (InvalidSignature, ValueError):
        log_security_event(db, "DEVICE_SIGNATURE_INVALID", f"Invalid signature from device {d.name}", "HIGH",
                           ip_address=ip, org_id=d.organization_id, metadata={"device_id": d.id})
        alert_service.raise_alert(db, "DEVICE_SIGNATURE_INVALID", "HIGH", "device", d.id,
                                  f"Device {d.name} sent a request with an invalid signature", org_id=d.organization_id)
        raise bad from None
    db.add(DeviceNonce(device_id=d.id, nonce=nonce))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        log_security_event(db, "DEVICE_REQUEST_REPLAY", f"Replayed request from device {d.name}", "HIGH",
                           ip_address=ip, org_id=d.organization_id, metadata={"device_id": d.id})
        raise ApiError(401, "REPLAYED_REQUEST", "This device request was already processed") from None
    d.last_seen_at, d.last_ip = datetime.utcnow(), ip[:45]
    db.commit()
    return d


def heartbeat(db: Session, d: Device, body: dict[str, Any]) -> dict[str, Any]:
    d.software_version = str(body.get("software_version") or d.software_version or "")[:40]
    health = body.get("health")
    if isinstance(health, dict):
        d.health_json = json.dumps({k: health[k] for k in list(health)[:20]}, default=str)[:4000]
    now = datetime.utcnow()
    for cam in body.get("cameras") or []:
        c = db.query(Camera).filter(Camera.id == str(cam.get("camera_id")), Camera.device_id == d.id).first()
        if c is not None and c.status == "ACTIVE":
            c.last_heartbeat_at, c.last_heartbeat_by = now, f"device:{d.name}"[:64]
    db.commit()
    return {"status": d.status, "server_time": now.isoformat() + "Z", "offline_enabled": d.offline_enabled}


def rotate_key(db: Session, d: Device, new_public_key_pem: str, ip: str) -> Device:
    pub = _load_public_key(new_public_key_pem)
    d.public_key_pem, d.key_fingerprint, d.key_rotated_at = new_public_key_pem, _fingerprint(pub), datetime.utcnow()
    db.commit()
    audit(db, None, "DEVICE_KEY_ROTATED", "device", d.id, {"fingerprint": d.key_fingerprint[:16]}, ip,
          org_id=d.organization_id)
    return d


# ── cameras ─────────────────────────────────────────────────────────────────

def camera_status(c: Camera) -> str:
    if c.status != "ACTIVE":
        return "DISABLED"
    if c.last_heartbeat_at is None:
        return "NEVER_SEEN"
    return "ONLINE" if (datetime.utcnow() - c.last_heartbeat_at).total_seconds() <= CAMERA_ONLINE_WINDOW_S else "OFFLINE"


def camera_view(c: Camera) -> dict[str, Any]:
    return {"id": c.id, "label": c.label, "location": c.location, "hardware_label": c.hardware_label,
            "capabilities": json.loads(c.capabilities_json) if c.capabilities_json else None,
            "status": c.status, "connection": camera_status(c), "device_id": c.device_id,
            "last_heartbeat_at": c.last_heartbeat_at, "last_heartbeat_by": c.last_heartbeat_by,
            "last_verification_at": c.last_verification_at, "created_at": c.created_at, "disabled_at": c.disabled_at}


def load_camera(db: Session, actor: Actor, camera_id: str) -> Camera:
    c = db.query(Camera).filter(Camera.id == camera_id, Camera.organization_id == actor.org_id).first()
    if c is None:
        raise ApiError(404, "NOT_FOUND", "Camera not found")
    return c


def _check_device(db: Session, actor: Actor, device_id: Optional[str]) -> None:
    if device_id:
        load_device(db, actor, device_id)


# ── offline kiosks ──────────────────────────────────────────────────────────

def set_offline_subjects(db: Session, actor: Actor, device_id: str, user_ids: list[str], ip: str) -> Device:
    d = load_device(db, actor, device_id)
    if d.kind != DeviceKind.KIOSK or not d.offline_enabled:
        raise ApiError(409, "OFFLINE_NOT_ENABLED", "Offline verification is not enabled for this device")
    policy = get_active_policy(db, actor.org_id)
    ids = list(dict.fromkeys(user_ids))
    if len(ids) > int(policy.offline_max_subjects):
        raise ApiError(422, "TOO_MANY_SUBJECTS", f"At most {policy.offline_max_subjects} subjects per kiosk")
    from backend.services.enrolment_service import get_enrolment
    for uid in ids:
        enr = get_enrolment(db, actor.org_id, uid)
        if membership_role(db, actor.org_id, uid) is None or enr is None or not enr.is_usable:
            raise ApiError(422, "SUBJECT_NOT_ELIGIBLE", "Every subject must be an enrolled member of this organisation")
    db.query(OfflineSubject).filter(OfflineSubject.device_id == d.id).delete(synchronize_session=False)
    for uid in ids:
        db.add(OfflineSubject(device_id=d.id, user_id=uid, added_by_user_id=actor.id))
    db.commit()
    audit(db, actor.id, "OFFLINE_SUBJECTS_SET", "device", d.id, {"count": len(ids)}, ip, org_id=actor.org_id)
    return d


def build_bundle(db: Session, d: Device) -> dict[str, Any]:
    if d.kind != DeviceKind.KIOSK or not d.offline_enabled:
        raise ApiError(409, "OFFLINE_NOT_ENABLED", "Offline verification is not enabled for this device")
    from backend.services.biometric.templates import decrypt_vectors
    from backend.services.enrolment_service import get_enrolment
    from backend.services.integration_service import active_signing_key, sign
    policy = get_active_policy(db, d.organization_id)
    hours = min(d.offline_validity_hours, int(policy.offline_validity_hours), settings.offline_max_validity_hours)
    now = datetime.utcnow().replace(microsecond=0)
    valid_until = now + timedelta(hours=hours)
    subjects = []
    for s, u in (db.query(OfflineSubject, User).join(User, User.id == OfflineSubject.user_id)
                 .filter(OfflineSubject.device_id == d.id).all()):
        enr = get_enrolment(db, d.organization_id, u.id)
        if not u.is_active or enr is None or not enr.is_usable or membership_role(db, d.organization_id, u.id) is None:
            continue                       # revoked / withdrawn / removed since authorisation
        t = decrypt_vectors(enr.embedding_encrypted)
        subjects.append({"user_id": u.id, "display_name": u.full_name, "model": t.model, "dim": t.dim,
                         "template": _b64(t.vector.astype("float32").tobytes())})
    bundle_id = str(uuid.uuid4())
    plaintext = json.dumps({"bundle_id": bundle_id, "device_id": d.id, "organization_id": d.organization_id,
                            "issued_at": now.isoformat() + "Z", "valid_until": valid_until.isoformat() + "Z",
                            "policy": json.loads(policy.to_json()), "subjects": subjects}).encode()
    device_pub = serialization.load_pem_public_key(d.public_key_pem.encode())
    eph = ec.generate_private_key(ec.SECP256R1())
    shared = eph.exchange(ec.ECDH(), device_pub)
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=bundle_id.encode(), info=BUNDLE_INFO).derive(shared)
    nonce = os.urandom(12)
    aad = f"{bundle_id}|{d.id}".encode()
    ct = AESGCM(key).encrypt(nonce, plaintext, aad)
    epk = eph.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    digest = hashlib.sha256(ct).hexdigest()
    signature = sign(db, {"bundle_id": bundle_id, "device_id": d.id, "sha256": digest,
                          "valid_until": valid_until.isoformat() + "Z", "iss": "faceverification-offline"})
    db.query(OfflineBundle).filter(OfflineBundle.device_id == d.id, OfflineBundle.status == "ISSUED").update(
        {"status": "SUPERSEDED"}, synchronize_session=False)
    db.add(OfflineBundle(id=bundle_id, device_id=d.id, organization_id=d.organization_id, issued_at=now,
                         valid_until=valid_until, subject_count=len(subjects), policy_version=policy.version,
                         payload_sha256=digest, status="ISSUED"))
    db.commit()
    audit(db, None, "OFFLINE_BUNDLE_ISSUED", "device", d.id, {"bundle_id": bundle_id, "subjects": len(subjects),
                                                            "valid_until": valid_until.isoformat()}, None,
          org_id=d.organization_id)
    return {"bundle_id": bundle_id, "valid_until": valid_until.isoformat() + "Z", "subject_count": len(subjects),
            "epk": epk, "nonce": _b64(nonce), "ciphertext": _b64(ct), "signature": signature,
            "signing_kid": active_signing_key(db).id}


def _chain(prev_hash: str, ev: dict[str, Any]) -> str:
    fields = {k: ev.get(k) for k in ("event_id", "seq", "type", "occurred_at", "subject_user_id", "outcome",
                                     "reason_code", "payload", "bundle_id")}
    return hashlib.sha256((prev_hash or "").encode() + json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()


def sync_events(db: Session, d: Device, events: list[dict[str, Any]], ip: str) -> dict[str, Any]:
    """Idempotent upload of kiosk audit events. (device_id, event_id) is unique; duplicates are acknowledged."""
    acked, duplicates, rejected = [], [], []
    broken = 0
    last = (db.query(OfflineEvent).filter(OfflineEvent.device_id == d.id).order_by(OfflineEvent.seq.desc()).first())
    expected_prev, expected_seq = (last.chain_hash, last.seq) if last else ("", 0)
    for ev in sorted(events, key=lambda e: int(e.get("seq", 0)))[:500]:
        eid = str(ev.get("event_id") or "")[:36]
        try:
            uuid.UUID(eid)
        except ValueError:
            rejected.append({"event_id": eid, "reason": "INVALID_ID"})
            continue
        if db.query(OfflineEvent).filter(OfflineEvent.device_id == d.id, OfflineEvent.event_id == eid).first():
            duplicates.append(eid)
            continue
        integrity = "OK"
        if _chain(ev.get("prev_hash") or "", ev) != ev.get("chain_hash"):
            integrity = "BROKEN_CHAIN"
        elif int(ev.get("seq", 0)) == expected_seq + 1 and (ev.get("prev_hash") or "") != expected_prev:
            integrity = "BROKEN_CHAIN"
        bundle = db.query(OfflineBundle).filter(OfflineBundle.id == ev.get("bundle_id"),
                                                OfflineBundle.device_id == d.id).first()
        try:
            occurred = datetime.fromisoformat(str(ev.get("occurred_at", "")).replace("Z", ""))
        except ValueError:
            rejected.append({"event_id": eid, "reason": "INVALID_TIME"})
            continue
        if integrity == "OK" and (bundle is None or occurred > bundle.valid_until):
            integrity = "LATE"
        subject = ev.get("subject_user_id")
        if subject and membership_role(db, d.organization_id, subject) is None:
            subject = None
        row = OfflineEvent(device_id=d.id, organization_id=d.organization_id, event_id=eid, seq=int(ev.get("seq", 0)),
                           event_type=str(ev.get("type", ""))[:32], occurred_at=occurred,
                           bundle_id=bundle.id if bundle else None, subject_user_id=subject,
                           outcome=(ev.get("outcome") or None) and str(ev["outcome"])[:32],
                           reason_code=(ev.get("reason_code") or None) and str(ev["reason_code"])[:48],
                           payload_json=json.dumps(ev.get("payload"))[:4000] if ev.get("payload") else None,
                           prev_hash=(ev.get("prev_hash") or "")[:64], chain_hash=str(ev.get("chain_hash", ""))[:64],
                           integrity=integrity)
        db.add(row)
        try:
            db.commit()
        except IntegrityError:          # concurrent duplicate upload
            db.rollback()
            duplicates.append(eid)
            continue
        if integrity != "OK":
            broken += 1
        acked.append(eid)
        expected_prev, expected_seq = row.chain_hash, row.seq
    if broken:
        alert_service.raise_alert(db, "OFFLINE_AUDIT_INTEGRITY", "HIGH", "device", d.id,
                                  f"{broken} offline event(s) from {d.name} failed integrity or validity checks",
                                  org_id=d.organization_id)
    if acked:
        audit(db, None, "OFFLINE_EVENTS_SYNCED", "device", d.id, {"acked": len(acked), "duplicates": len(duplicates),
                                                                 "integrity_issues": broken}, ip,
              org_id=d.organization_id)
        bus.publish("devices", "offline_synced", {"id": d.id, "events": len(acked)}, org_id=d.organization_id)
    return {"acked": acked, "duplicates": duplicates, "rejected": rejected, "integrity_issues": broken,
            "server_time": datetime.utcnow().isoformat() + "Z"}


def maintenance_task(db: Session, now: datetime, _policies: dict) -> dict[str, int]:
    n = db.query(DeviceNonce).filter(DeviceNonce.created_at < now - timedelta(minutes=15)).delete(synchronize_session=False)
    return {"device_nonces_deleted": n}
