"""Integration API: clients, scoped credentials, hosted verification sessions, signed assertions, webhooks.

* Credentials look like `fvk_<key_id>_<secret>`. Only an HMAC of the secret is stored; the
  secret is shown once. Rotation issues a new credential and lets old ones expire after an
  optional grace period. Credentials are scoped and rate-limited per client.
* Sessions created through the API are hosted links (see pairing_service, kind HOSTED):
  the client hands the link to its end user, who verifies on their own device.
* Results are delivered as ES256-signed JWT assertions (iss, aud, sub, outcome, iat, exp,
  jti, policy version …). Public keys are published as a JWKS. An assertion can be
  introspected once (replay protection on the server side as well as by `jti`).
* Webhooks carry `FV-Signature: t=<unix>,v1=<hex HMAC-SHA256(secret, "<t>.<body>")>` and are
  retried with exponential back-off. Raw biometric data never leaves through this API.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.parse import urlparse

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from jose import jwt
from jose.exceptions import JWTError
from sqlalchemy.orm import Session

from backend.config import settings
from backend.core import crypto
from backend.core.deps import Actor, membership_role
from backend.core.errors import ApiError
from backend.models.integrations import (
    ApiCredential, IdempotencyRecord, IntegrationClient, SigningKey, VerificationAssertion, WebhookDelivery,
    WebhookEndpoint,
)
from backend.models.organization import Organization, OrgStatus
from backend.models.pairing import CapturePairing, PairingKind
from backend.models.user import User
from backend.models.verification import VerificationSession
from backend.services import alert_service
from backend.services.audit_service import audit, log_security_event

log = logging.getLogger(__name__)

SCOPES = ("sessions:write", "sessions:read", "assertions:introspect")
EVENTS = ("verification.completed", "verification.reviewed", "ping")
WEBHOOK_BACKOFF_MINUTES = (1, 5, 30, 120, 360)
MAX_WEBHOOK_ATTEMPTS = 6
SIGNATURE_TOLERANCE_S = 300


# ── signing keys / assertions ───────────────────────────────────────────────

def active_signing_key(db: Session, commit: bool = True) -> SigningKey:
    """The current assertion/bundle signing key, created on first use. `commit=False` inside hooks."""
    key = db.query(SigningKey).filter(SigningKey.status == "ACTIVE").order_by(SigningKey.created_at.desc()).first()
    if key is not None:
        return key
    priv = ec.generate_private_key(ec.SECP256R1())
    pem = priv.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    pub = priv.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    key = SigningKey(id=str(uuid.uuid4()), alg="ES256", public_pem=pub.decode(), private_encrypted=crypto.seal(pem))
    db.add(key)
    if commit:
        db.commit()
    else:
        db.flush()
    return key


def rotate_signing_key(db: Session) -> SigningKey:
    for k in db.query(SigningKey).filter(SigningKey.status == "ACTIVE"):
        k.status, k.retired_at = "RETIRED", datetime.utcnow()
    db.commit()
    return active_signing_key(db)


def jwks(db: Session) -> dict[str, Any]:
    active_signing_key(db)
    keys = []
    cutoff = datetime.utcnow() - timedelta(days=7)        # retired keys stay published for a week
    for k in db.query(SigningKey).filter((SigningKey.status == "ACTIVE") | (SigningKey.retired_at >= cutoff)):
        nums = serialization.load_pem_public_key(k.public_pem.encode()).public_numbers()
        from webauthn.helpers import bytes_to_base64url
        keys.append({"kty": "EC", "crv": "P-256", "alg": "ES256", "use": "sig", "kid": k.id,
                     "x": bytes_to_base64url(nums.x.to_bytes(32, "big")),
                     "y": bytes_to_base64url(nums.y.to_bytes(32, "big"))})
    return {"keys": keys}


def issuer() -> str:
    return f"{settings.public_base_url.rstrip('/')}/api/v1/integrations"


def sign(db: Session, claims: dict[str, Any], commit: bool = True) -> str:
    key = active_signing_key(db, commit=commit)
    return jwt.encode(claims, crypto.unseal(key.private_encrypted).decode(), algorithm="ES256",
                      headers={"kid": key.id, "typ": "fv-assertion+jwt"})


def _assertion_claims(db: Session, a: VerificationAssertion, s: VerificationSession, p: CapturePairing) -> dict:
    return {"iss": issuer(), "aud": a.client_id, "sub": a.subject, "jti": a.id,
            "iat": _ts(a.issued_at),
            "exp": _ts(a.expires_at), "outcome": s.final_outcome, "reason_code": s.reason_code,
            "review_status": s.review_status, "verification_id": p.id, "external_ref": p.external_ref,
            "liveness": "passed" if s.final_outcome in ("VERIFIED", "NOT_MATCHED") else (
                "failed" if s.final_outcome == "LIVENESS_FAILED" else "not_completed"),
            "policy_version": s.policy_version, "org": s.organization_id, "channel": s.channel}


def _ts(dt: datetime) -> int:
    import calendar
    return calendar.timegm(dt.utctimetuple())


def assertion_for(db: Session, p: CapturePairing, commit: bool = True) -> Optional[str]:
    """(Re-)sign the stored assertion claims for a finished hosted session (same jti, iat and exp)."""
    a = (db.query(VerificationAssertion).filter(VerificationAssertion.session_id == p.session_id)
         .order_by(VerificationAssertion.issued_at.desc()).first()) if p.session_id else None
    if a is None:
        return None
    s = db.query(VerificationSession).filter(VerificationSession.id == a.session_id).first()
    return sign(db, _assertion_claims(db, a, s, p), commit=commit)


def introspect(db: Session, client: IntegrationClient, token: str) -> dict[str, Any]:
    try:
        header = jwt.get_unverified_header(token)
        key = db.query(SigningKey).filter(SigningKey.id == header.get("kid")).first()
        if key is None:
            return {"active": False, "reason": "unknown_key"}
        claims = jwt.decode(token, key.public_pem, algorithms=["ES256"], audience=client.id, issuer=issuer())
    except JWTError as exc:
        return {"active": False, "reason": "invalid", "detail": str(exc)[:120]}
    n = (db.query(VerificationAssertion)
         .filter(VerificationAssertion.id == claims["jti"], VerificationAssertion.client_id == client.id,
                 VerificationAssertion.consumed_at.is_(None))
         .update({"consumed_at": datetime.utcnow()}, synchronize_session=False))
    db.commit()
    if n != 1:
        return {"active": False, "reason": "already_used", "jti": claims["jti"]}
    return {"active": True, "claims": claims}


# ── clients and credentials ─────────────────────────────────────────────────

def _make_credential(db: Session, client: IntegrationClient, actor_id: Optional[str]) -> tuple[ApiCredential, str]:
    key_id = crypto.token(9).replace("-", "a").replace("_", "b")[:12].lower()
    secret = crypto.token(32)
    cred = ApiCredential(client_id=client.id, key_id=key_id, secret_hash=crypto.keyed_hash(secret, "apikey"),
                         created_by_user_id=actor_id)
    db.add(cred)
    db.flush()
    return cred, f"fvk_{key_id}_{secret}"


def create_client(db: Session, actor: Actor, name: str, description: str, scopes: list[str],
                  rate_limit: int, ip: str) -> tuple[IntegrationClient, str]:
    bad = [s for s in scopes if s not in SCOPES]
    if bad or not scopes:
        raise ApiError(422, "INVALID_SCOPES", f"Scopes must be a non-empty subset of {', '.join(SCOPES)}")
    client = IntegrationClient(organization_id=actor.org_id, name=name.strip()[:120], description=description[:2000],
                               scopes=",".join(sorted(set(scopes))), rate_limit_per_minute=rate_limit,
                               created_by_user_id=actor.id)
    db.add(client)
    db.flush()
    cred, raw = _make_credential(db, client, actor.id)
    db.commit()
    audit(db, actor.id, "INTEGRATION_CLIENT_CREATED", "integration_client", client.id,
          {"name": client.name, "scopes": client.scopes, "key_id": cred.key_id}, ip, org_id=actor.org_id)
    log_security_event(db, "API_CREDENTIAL_CREATED", f"Credential {cred.key_id} issued for client {client.name}",
                       "WARNING", user_id=actor.id, ip_address=ip, org_id=actor.org_id)
    return client, raw


def load_client(db: Session, actor: Actor, client_id: str) -> IntegrationClient:
    c = (db.query(IntegrationClient)
         .filter(IntegrationClient.id == client_id, IntegrationClient.organization_id == actor.org_id).first())
    if c is None:
        raise ApiError(404, "NOT_FOUND", "Integration client not found")
    return c


def rotate_credential(db: Session, actor: Actor, client_id: str, grace_hours: int, ip: str) -> tuple[ApiCredential, str]:
    client = load_client(db, actor, client_id)
    if client.status != "ACTIVE":
        raise ApiError(409, "CLIENT_REVOKED", "This client is revoked")
    expire_at = datetime.utcnow() + timedelta(hours=grace_hours)
    for c in db.query(ApiCredential).filter(ApiCredential.client_id == client.id, ApiCredential.revoked_at.is_(None)):
        if c.expires_at is None or c.expires_at > expire_at:
            c.expires_at = expire_at
    cred, raw = _make_credential(db, client, actor.id)
    db.commit()
    audit(db, actor.id, "API_CREDENTIAL_ROTATED", "integration_client", client.id,
          {"new_key_id": cred.key_id, "grace_hours": grace_hours}, ip, org_id=actor.org_id)
    log_security_event(db, "API_CREDENTIAL_ROTATED", f"Credential rotated for client {client.name}", "WARNING",
                       user_id=actor.id, ip_address=ip, org_id=actor.org_id)
    return cred, raw


def revoke_credential(db: Session, actor: Actor, client_id: str, credential_id: str, ip: str) -> ApiCredential:
    client = load_client(db, actor, client_id)
    c = db.query(ApiCredential).filter(ApiCredential.id == credential_id, ApiCredential.client_id == client.id).first()
    if c is None:
        raise ApiError(404, "NOT_FOUND", "Credential not found")
    if c.revoked_at is None:
        c.revoked_at = datetime.utcnow()
        db.commit()
        audit(db, actor.id, "API_CREDENTIAL_REVOKED", "integration_client", client.id, {"key_id": c.key_id}, ip,
              org_id=actor.org_id)
    return c


def revoke_client(db: Session, actor: Actor, client_id: str, ip: str) -> IntegrationClient:
    client = load_client(db, actor, client_id)
    now = datetime.utcnow()
    client.status, client.revoked_at = "REVOKED", now
    db.query(ApiCredential).filter(ApiCredential.client_id == client.id, ApiCredential.revoked_at.is_(None)).update(
        {"revoked_at": now}, synchronize_session=False)
    db.query(WebhookEndpoint).filter(WebhookEndpoint.client_id == client.id).update({"status": "DISABLED"},
                                                                                     synchronize_session=False)
    db.commit()
    audit(db, actor.id, "INTEGRATION_CLIENT_REVOKED", "integration_client", client.id, None, ip, org_id=actor.org_id)
    return client


# ── API authentication and rate limiting ────────────────────────────────────

@dataclass
class ApiPrincipal:
    client: IntegrationClient
    credential: ApiCredential
    org_id: str
    scopes: frozenset

    def require(self, scope: str) -> None:
        if scope not in self.scopes:
            raise ApiError(403, "INSUFFICIENT_SCOPE", f"This credential lacks the '{scope}' scope")


class _RateLimiter:
    """Per-client sliding one-minute window (per API process)."""

    def __init__(self) -> None:
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= limit:
                return False, max(1, int(60 - (now - q[0])))
            q.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


rate_limiter = _RateLimiter()


def authenticate(db: Session, raw: str, ip: str) -> ApiPrincipal:
    unauthorized = ApiError(401, "INVALID_API_KEY", "Invalid or revoked API credential",
                            headers={"WWW-Authenticate": "Bearer"})
    if not raw or not raw.startswith("fvk_") or raw.count("_") < 2 or len(raw) > 200:
        raise unauthorized
    _, key_id, secret = raw.split("_", 2)
    cred = db.query(ApiCredential).filter(ApiCredential.key_id == key_id).first()
    now = datetime.utcnow()
    if cred is None or not crypto.constant_time_equals(cred.secret_hash, crypto.keyed_hash(secret, "apikey")):
        if cred is not None:
            client = db.query(IntegrationClient).filter(IntegrationClient.id == cred.client_id).first()
            log_security_event(db, "API_KEY_INVALID_SECRET", f"Wrong secret for key {key_id}", "HIGH", ip_address=ip,
                               org_id=client.organization_id if client else None)
            if client:
                alert_service.raise_alert(db, "API_CREDENTIAL_ABUSE", "HIGH", "api_key", key_id,
                                          f"Invalid secret presented for API key {key_id} from {ip}",
                                          org_id=client.organization_id)
        raise unauthorized
    if cred.revoked_at is not None or (cred.expires_at is not None and cred.expires_at <= now):
        raise unauthorized
    client = db.query(IntegrationClient).filter(IntegrationClient.id == cred.client_id).first()
    org = db.query(Organization).filter(Organization.id == client.organization_id).first() if client else None
    if client is None or client.status != "ACTIVE" or org is None or org.status != OrgStatus.ACTIVE:
        raise unauthorized
    ok, retry = rate_limiter.allow(client.id, min(client.rate_limit_per_minute, settings.api_rate_limit_per_minute * 10))
    if not ok:
        raise ApiError(429, "RATE_LIMITED", "Rate limit exceeded", headers={"Retry-After": str(retry)})
    if cred.last_used_at is None or (now - cred.last_used_at).total_seconds() > 30:
        cred.last_used_at = now
        db.commit()
    return ApiPrincipal(client=client, credential=cred, org_id=client.organization_id,
                        scopes=frozenset(s for s in client.scopes.split(",") if s))


# ── idempotency ─────────────────────────────────────────────────────────────

def idempotent(db: Session, principal: ApiPrincipal, key: Optional[str], body: dict) -> Optional[tuple[int, dict]]:
    if not key:
        return None
    if len(key) > 128:
        raise ApiError(422, "INVALID_IDEMPOTENCY_KEY", "Idempotency-Key is at most 128 characters")
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    rec = (db.query(IdempotencyRecord)
           .filter(IdempotencyRecord.client_id == principal.client.id, IdempotencyRecord.key == key).first())
    if rec is None:
        return None
    if rec.request_hash != h:
        raise ApiError(409, "IDEMPOTENCY_KEY_REUSED", "This Idempotency-Key was used with a different request body")
    return rec.status_code, json.loads(rec.response_json)


def remember(db: Session, principal: ApiPrincipal, key: Optional[str], body: dict, status: int, response: dict) -> None:
    if not key:
        return
    db.add(IdempotencyRecord(client_id=principal.client.id, key=key,
                             request_hash=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                             status_code=status, response_json=json.dumps(response, default=str)))
    db.commit()


# ── hosted sessions ─────────────────────────────────────────────────────────

def create_session(db: Session, principal: ApiPrincipal, user_id: Optional[str], email: Optional[str],
                   external_ref: Optional[str], expires_in: int) -> dict[str, Any]:
    principal.require("sessions:write")
    q = db.query(User)
    target = q.filter(User.id == user_id).first() if user_id else (
        q.filter(User.email == (email or "").strip().lower()).first() if email else None)
    if target is None or not target.is_active or membership_role(db, principal.org_id, target.id) is None:
        raise ApiError(404, "SUBJECT_NOT_FOUND", "Subject not found in this organisation")
    from backend.services import capture_service
    from backend.services.enrolment_service import get_enrolment
    try:
        capture_service.check_verifiable(get_enrolment(db, principal.org_id, target.id))
    except capture_service.CaptureError as exc:
        raise ApiError(409, exc.code, exc.message) from exc
    from backend.services import pairing_service
    p, raw = pairing_service.create_hosted(db, principal.org_id, principal.client.id, target, external_ref, expires_in)
    db.commit()
    audit(db, None, "API_SESSION_CREATED", "pairing", p.id, {"client": principal.client.id}, None,
          org_id=principal.org_id)
    return {**session_view(db, p), "verification_url": pairing_service._url(raw)}


def load_api_session(db: Session, principal: ApiPrincipal, verification_id: str) -> CapturePairing:
    p = (db.query(CapturePairing)
         .filter(CapturePairing.id == verification_id, CapturePairing.integration_client_id == principal.client.id)
         .first())
    if p is None:
        raise ApiError(404, "NOT_FOUND", "Verification not found")
    return p


def session_view(db: Session, p: CapturePairing, expire: bool = True) -> dict[str, Any]:
    from backend.services import pairing_service
    if expire:
        pairing_service._expire_if_needed(db, p)
    s = db.query(VerificationSession).filter(VerificationSession.id == p.session_id).first() if p.session_id else None
    return {"id": p.id, "status": p.status, "external_ref": p.external_ref, "subject_id": p.target_user_id,
            "created_at": p.created_at, "expires_at": p.expires_at, "completed_at": p.completed_at,
            "outcome": p.result_outcome, "review_status": s.review_status if s else None,
            "attempts": p.sessions_started}


def on_session_closed(db: Session, s: VerificationSession, _decision=None) -> None:
    """Terminal hook: for a finished hosted session, issue the assertion and queue webhooks."""
    if not s.pairing_id or s.channel != "HOSTED":
        return
    p = db.query(CapturePairing).filter(CapturePairing.id == s.pairing_id).first()
    if p is None or p.kind != PairingKind.HOSTED or p.integration_client_id is None or p.status != "COMPLETED":
        return
    if db.query(VerificationAssertion).filter(VerificationAssertion.session_id == s.id).first():
        return
    now = datetime.utcnow().replace(microsecond=0)
    a = VerificationAssertion(id=str(uuid.uuid4()), organization_id=s.organization_id,
                              client_id=p.integration_client_id, session_id=s.id,
                              subject=p.external_ref or p.target_user_id, outcome=s.final_outcome,
                              kid=active_signing_key(db, commit=False).id, issued_at=now,
                              expires_at=now + timedelta(seconds=settings.assertion_ttl_seconds))
    db.add(a)
    db.flush()
    enqueue(db, p, "verification.completed", commit=False)


def on_review_decided(db: Session, s: VerificationSession, review) -> None:
    if not s.pairing_id or s.channel != "HOSTED":
        return
    p = db.query(CapturePairing).filter(CapturePairing.id == s.pairing_id).first()
    if p is not None and p.integration_client_id:
        enqueue(db, p, "verification.reviewed", {"review_status": review.status})
        db.commit()


# ── webhooks ────────────────────────────────────────────────────────────────

def validate_webhook_url(url: str) -> None:
    u = urlparse(url)
    if u.scheme not in ("https", "http") or not u.hostname:
        raise ApiError(422, "INVALID_URL", "Webhook URL must be an absolute http(s) URL")
    if settings.webhook_allow_private_targets:
        return
    if u.scheme != "https":
        raise ApiError(422, "HTTPS_REQUIRED", "Webhook URL must use HTTPS")
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or 443)
    except socket.gaierror as exc:
        raise ApiError(422, "UNRESOLVABLE_HOST", "Webhook host does not resolve") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ApiError(422, "PRIVATE_TARGET", "Webhook URL must not point to a private or local address")


def create_webhook(db: Session, actor: Actor, client_id: str, url: str, events: list[str], ip: str) -> tuple[WebhookEndpoint, str]:
    client = load_client(db, actor, client_id)
    bad = [e for e in events if e not in EVENTS]
    if bad or not events:
        raise ApiError(422, "INVALID_EVENTS", f"Events must be a subset of {', '.join(EVENTS)}")
    validate_webhook_url(url)
    secret = "whsec_" + crypto.token(32)
    ep = WebhookEndpoint(organization_id=actor.org_id, client_id=client.id, url=url[:500],
                         secret_encrypted=crypto.seal(secret.encode()), events=",".join(sorted(set(events))))
    db.add(ep)
    db.commit()
    audit(db, actor.id, "WEBHOOK_CREATED", "webhook", ep.id, {"url": url, "events": ep.events}, ip, org_id=actor.org_id)
    return ep, secret


def load_webhook(db: Session, actor: Actor, webhook_id: str) -> WebhookEndpoint:
    ep = (db.query(WebhookEndpoint)
          .filter(WebhookEndpoint.id == webhook_id, WebhookEndpoint.organization_id == actor.org_id).first())
    if ep is None:
        raise ApiError(404, "NOT_FOUND", "Webhook not found")
    return ep


def rotate_webhook_secret(db: Session, actor: Actor, webhook_id: str, ip: str) -> str:
    ep = load_webhook(db, actor, webhook_id)
    secret = "whsec_" + crypto.token(32)
    ep.secret_encrypted = crypto.seal(secret.encode())
    ep.secret_rotated_at = datetime.utcnow()
    db.commit()
    audit(db, actor.id, "WEBHOOK_SECRET_ROTATED", "webhook", ep.id, None, ip, org_id=actor.org_id)
    return secret


def signature_header(secret: str, body: bytes, ts: Optional[int] = None) -> str:
    ts = int(time.time()) if ts is None else ts
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def verify_signature(secret: str, body: bytes, header: str, tolerance: int = SIGNATURE_TOLERANCE_S) -> bool:
    try:
        parts = dict(x.split("=", 1) for x in header.split(","))
        ts = int(parts["t"])
    except (ValueError, KeyError):
        return False
    if abs(time.time() - ts) > tolerance:
        return False
    expected = signature_header(secret, body, ts).split("v1=", 1)[1]
    return hmac.compare_digest(expected, parts.get("v1", ""))


def enqueue(db: Session, p: CapturePairing, event_type: str, extra: Optional[dict] = None, commit: bool = True) -> int:
    endpoints = (db.query(WebhookEndpoint)
                 .filter(WebhookEndpoint.client_id == p.integration_client_id, WebhookEndpoint.status == "ACTIVE").all())
    n = 0
    for ep in endpoints:
        if event_type not in ep.events.split(","):
            continue
        event_id = str(uuid.uuid4())
        payload = {"id": event_id, "type": event_type, "created_at": datetime.utcnow().isoformat() + "Z",
                   "data": {**session_view(db, p, expire=False), **(extra or {}),
                            "assertion": assertion_for(db, p, commit=commit)}}
        db.add(WebhookDelivery(endpoint_id=ep.id, organization_id=ep.organization_id, event_id=event_id,
                               event_type=event_type, payload_json=json.dumps(payload, default=str),
                               status="PENDING", next_attempt_at=datetime.utcnow()))
        n += 1
    return n


def send_ping(db: Session, actor: Actor, webhook_id: str) -> WebhookDelivery:
    ep = load_webhook(db, actor, webhook_id)
    event_id = str(uuid.uuid4())
    d = WebhookDelivery(endpoint_id=ep.id, organization_id=ep.organization_id, event_id=event_id, event_type="ping",
                        payload_json=json.dumps({"id": event_id, "type": "ping",
                                                 "created_at": datetime.utcnow().isoformat() + "Z", "data": {}}),
                        status="PENDING", next_attempt_at=datetime.utcnow())
    db.add(d)
    db.commit()
    deliver(db, d)
    return d


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):   # never follow redirects (SSRF hardening)
        return None


_NO_REDIRECT = urllib.request.build_opener(_NoRedirect)


def deliver(db: Session, d: WebhookDelivery) -> None:
    ep = db.query(WebhookEndpoint).filter(WebhookEndpoint.id == d.endpoint_id).first()
    if ep is None or ep.status != "ACTIVE":
        d.status, d.last_error, d.next_attempt_at = "FAILED", "Endpoint disabled", None
        db.commit()
        return
    body = d.payload_json.encode()
    d.attempts += 1
    try:
        validate_webhook_url(ep.url)
        req = urllib.request.Request(ep.url, data=body, method="POST", headers={
            "Content-Type": "application/json", "User-Agent": "FaceVerification-Webhooks/4.0",
            "FV-Event-Id": d.event_id, "FV-Event-Type": d.event_type,
            "FV-Signature": signature_header(crypto.unseal(ep.secret_encrypted).decode(), body)})
        try:
            with _NO_REDIRECT.open(req, timeout=settings.webhook_timeout_seconds) as resp:
                status = resp.status
        except urllib.error.HTTPError as exc:      # non-2xx responses (and refused redirects)
            status = exc.code
        d.last_status_code = status
        ok = 200 <= status < 300
        d.last_error = None if ok else f"HTTP {status}"
    except (urllib.error.URLError, OSError, ApiError) as exc:
        ok = False
        d.last_status_code = None
        d.last_error = (getattr(exc, "message", None) or f"{type(exc).__name__}: {exc}")[:300]
    if ok:
        d.status, d.delivered_at, d.next_attempt_at = "DELIVERED", datetime.utcnow(), None
    elif d.attempts >= MAX_WEBHOOK_ATTEMPTS:
        d.status, d.next_attempt_at = "FAILED", None
    else:
        d.next_attempt_at = datetime.utcnow() + timedelta(
            minutes=WEBHOOK_BACKOFF_MINUTES[min(d.attempts - 1, len(WEBHOOK_BACKOFF_MINUTES) - 1)])
    db.commit()


def deliver_due(db: Session, now: Optional[datetime] = None, limit: int = 50) -> dict[str, int]:
    now = now or datetime.utcnow()
    rows = (db.query(WebhookDelivery)
            .filter(WebhookDelivery.status == "PENDING", WebhookDelivery.next_attempt_at <= now)
            .order_by(WebhookDelivery.next_attempt_at).limit(limit).all())
    for d in rows:
        deliver(db, d)
    return {"webhooks_attempted": len(rows)}


def maintenance_task(db: Session, now: datetime, _policies: dict) -> dict[str, int]:
    out = deliver_due(db, now)
    out["idempotency_records_deleted"] = (db.query(IdempotencyRecord)
                                          .filter(IdempotencyRecord.created_at < now - timedelta(hours=24))
                                          .delete(synchronize_session=False))
    return out


def client_view(db: Session, c: IntegrationClient, detail: bool = False) -> dict[str, Any]:
    out = {"id": c.id, "name": c.name, "description": c.description, "scopes": [s for s in c.scopes.split(",") if s],
           "status": c.status, "rate_limit_per_minute": c.rate_limit_per_minute, "created_at": c.created_at,
           "revoked_at": c.revoked_at}
    now = datetime.utcnow()
    creds = db.query(ApiCredential).filter(ApiCredential.client_id == c.id).order_by(ApiCredential.created_at.desc()).all()
    out["active_credentials"] = sum(1 for k in creds if k.revoked_at is None and (k.expires_at is None or k.expires_at > now))
    if detail:
        out["credentials"] = [{"id": k.id, "key_id": k.key_id, "prefix": f"fvk_{k.key_id}_…", "created_at": k.created_at,
                               "expires_at": k.expires_at, "revoked_at": k.revoked_at, "last_used_at": k.last_used_at,
                               "state": "REVOKED" if k.revoked_at else (
                                   "EXPIRED" if k.expires_at and k.expires_at <= now else
                                   "EXPIRING" if k.expires_at else "ACTIVE")} for k in creds]
        eps = db.query(WebhookEndpoint).filter(WebhookEndpoint.client_id == c.id).all()
        out["webhooks"] = [{"id": e.id, "url": e.url, "events": e.events.split(","), "status": e.status,
                            "created_at": e.created_at, "secret_rotated_at": e.secret_rotated_at} for e in eps]
        ids = [e.id for e in eps]
        ds = (db.query(WebhookDelivery).filter(WebhookDelivery.endpoint_id.in_(ids))
              .order_by(WebhookDelivery.created_at.desc()).limit(50).all()) if ids else []
        out["deliveries"] = [{"id": d.id, "endpoint_id": d.endpoint_id, "event_id": d.event_id,
                              "event_type": d.event_type, "status": d.status, "attempts": d.attempts,
                              "last_status_code": d.last_status_code, "last_error": d.last_error,
                              "created_at": d.created_at, "delivered_at": d.delivered_at,
                              "next_attempt_at": d.next_attempt_at} for d in ds]
    return out
