"""Authentication factors: passkeys (WebAuthn), TOTP, email one-time codes, recovery codes,
multi-step sign-in (pending tokens) and step-up re-authentication.

Security properties:
  * Every WebAuthn ceremony uses a server-generated challenge stored in
    `auth_challenges`; it is single use (consumed with a conditional UPDATE),
    expires after 5 minutes and is bound to its purpose (and user, when known).
  * Origin and RP ID are verified by py_webauthn against server configuration,
    never against values supplied by the browser. User verification is required.
  * A sign counter that fails to advance (cloned authenticator) is rejected.
  * TOTP codes are accepted once: the matched time-step must be newer than the
    last accepted one.
  * Recovery codes and one-time codes are stored as keyed hashes and consumed once.
  * A face match is never a sign-in factor on its own. It can only follow a
    successful passkey assertion (FACE_PENDING), or be combined with an
    independent factor for high-risk step-up.
"""
from __future__ import annotations

import json
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import pyotp
from sqlalchemy.orm import Session
from webauthn import (
    generate_authentication_options, generate_registration_options, options_to_json,
    verify_authentication_response, verify_registration_response,
)
from webauthn.helpers import base64url_to_bytes, bytes_to_base64url
from webauthn.helpers.exceptions import InvalidAuthenticationResponse, InvalidRegistrationResponse
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria, PublicKeyCredentialDescriptor, ResidentKeyRequirement,
    UserVerificationRequirement,
)

from backend.config import settings
from backend.core import crypto
from backend.core.errors import ApiError
from backend.models.mfa import (
    AuthChallenge, ChallengeKind, FactorKind, MfaFactor, RecoveryCode, WebAuthnCredential,
)
from backend.models.organization import Membership, MembershipStatus
from backend.models.user import RoleEnum, User
from backend.services import alert_service, email_service, notification_service
from backend.services.audit_service import audit, log_security_event
from backend.services.auth_service import active_orgs, pick_org, verify_password
from backend.services.policy_service import get_active_policy

log = logging.getLogger(__name__)

CHALLENGE_TTL = timedelta(minutes=5)
PENDING_TTL = timedelta(minutes=5)
EMAIL_OTP_TTL = timedelta(minutes=10)
MAX_PENDING_ATTEMPTS = 5
MAX_FACE_SESSIONS = 3
RECOVERY_CODE_COUNT = 10
TOTP_ISSUER = "FaceVerification"


# ── challenges ──────────────────────────────────────────────────────────────

def _new_challenge(db: Session, kind: str, user_id: Optional[str] = None, org_id: Optional[str] = None,
                   challenge: Optional[str] = None, token: Optional[str] = None, data: Optional[dict] = None,
                   ttl: timedelta = CHALLENGE_TTL, ip: str = "") -> AuthChallenge:
    row = AuthChallenge(kind=kind, user_id=user_id, organization_id=org_id, challenge=challenge,
                        token_hash=crypto.keyed_hash(token, "authchallenge") if token else None,
                        data_json=json.dumps(data) if data else None, created_at=datetime.utcnow(),
                        expires_at=datetime.utcnow() + ttl, ip_address=ip or None)
    db.add(row)
    db.commit()
    return row


def _consume(db: Session, row: AuthChallenge) -> None:
    """Atomically mark a challenge used. A second use (replay or race) fails."""
    n = (db.query(AuthChallenge)
         .filter(AuthChallenge.id == row.id, AuthChallenge.consumed_at.is_(None),
                 AuthChallenge.expires_at >= datetime.utcnow())
         .update({"consumed_at": datetime.utcnow()}, synchronize_session=False))
    db.commit()
    if n != 1:
        raise ApiError(400, "CHALLENGE_USED", "This sign-in step has already been used or has expired")


def _load_challenge(db: Session, challenge_id: str, kind: str, user_id: Optional[str] = None) -> AuthChallenge:
    row = db.query(AuthChallenge).filter(AuthChallenge.id == challenge_id, AuthChallenge.kind == kind).first()
    if row is None or (user_id is not None and row.user_id != user_id):
        raise ApiError(400, "INVALID_CHALLENGE", "Unknown or mismatched challenge")
    if row.consumed_at is not None:
        raise ApiError(400, "CHALLENGE_USED", "This challenge has already been used")
    if row.expires_at < datetime.utcnow():
        raise ApiError(400, "CHALLENGE_EXPIRED", "This challenge has expired. Start again.")
    return row


def load_pending(db: Session, token: str, kind: str) -> AuthChallenge:
    if not token or len(token) > 200:
        raise ApiError(401, "INVALID_PENDING_TOKEN", "Sign-in step is not valid. Start again.")
    row = (db.query(AuthChallenge)
           .filter(AuthChallenge.token_hash == crypto.keyed_hash(token, "authchallenge"),
                   AuthChallenge.kind == kind).first())
    if row is None or row.consumed_at is not None or row.expires_at < datetime.utcnow():
        raise ApiError(401, "INVALID_PENDING_TOKEN", "Sign-in step expired or already used. Start again.")
    if row.attempts >= MAX_PENDING_ATTEMPTS:
        raise ApiError(429, "TOO_MANY_ATTEMPTS", "Too many failed attempts. Start again.")
    return row


def _data(row: AuthChallenge) -> dict[str, Any]:
    return json.loads(row.data_json) if row.data_json else {}


def _set_data(db: Session, row: AuthChallenge, data: dict[str, Any]) -> None:
    row.data_json = json.dumps(data)
    db.commit()


def _failed_attempt(db: Session, row: AuthChallenge, user: Optional[User], what: str, ip: str) -> None:
    row.attempts += 1
    db.commit()
    log_security_event(db, "MFA_FAILED", f"Failed {what} for {user.email if user else 'unknown user'}", "WARNING",
                       user_id=user.id if user else None, ip_address=ip)
    if user is not None and row.attempts >= 3:
        alert_service.raise_for_user(db, user.id, "MFA_FAILURES", "HIGH",
                                     f"{row.attempts} failed second-factor attempts in one sign-in", window_minutes=30)


# ── WebAuthn ────────────────────────────────────────────────────────────────

def _rp_origins() -> list[str]:
    return list(settings.webauthn_origins)


def user_handle(user: User) -> bytes:
    return user.id.encode()


def active_credentials(db: Session, user_id: str) -> list[WebAuthnCredential]:
    return (db.query(WebAuthnCredential)
            .filter(WebAuthnCredential.user_id == user_id, WebAuthnCredential.revoked_at.is_(None))
            .order_by(WebAuthnCredential.created_at).all())


def _descriptors(creds: list[WebAuthnCredential]) -> list[PublicKeyCredentialDescriptor]:
    return [PublicKeyCredentialDescriptor(id=base64url_to_bytes(c.credential_id)) for c in creds]


def registration_options(db: Session, user: User, ip: str) -> dict[str, Any]:
    challenge = secrets.token_bytes(32)
    opts = generate_registration_options(
        rp_id=settings.webauthn_rp_id, rp_name=settings.webauthn_rp_name, user_name=user.email,
        user_id=user_handle(user), user_display_name=user.full_name, challenge=challenge, timeout=300_000,
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.REQUIRED, user_verification=UserVerificationRequirement.REQUIRED),
        exclude_credentials=_descriptors(active_credentials(db, user.id)))
    row = _new_challenge(db, ChallengeKind.WEBAUTHN_REGISTER, user_id=user.id, challenge=bytes_to_base64url(challenge),
                         ip=ip)
    return {"challenge_id": row.id, "options": json.loads(options_to_json(opts))}


def register_credential(db: Session, user: User, challenge_id: str, credential: dict, name: str, ip: str,
                        org_id: Optional[str]) -> WebAuthnCredential:
    row = _load_challenge(db, challenge_id, ChallengeKind.WEBAUTHN_REGISTER, user.id)
    _consume(db, row)
    try:
        verified = verify_registration_response(
            credential=credential, expected_challenge=base64url_to_bytes(row.challenge),
            expected_rp_id=settings.webauthn_rp_id, expected_origin=_rp_origins(), require_user_verification=True)
    except InvalidRegistrationResponse as exc:
        log_security_event(db, "PASSKEY_REGISTRATION_FAILED", f"{user.email}: {str(exc)[:200]}", "WARNING",
                           user_id=user.id, ip_address=ip, org_id=org_id)
        raise ApiError(400, "REGISTRATION_INVALID", f"Passkey registration could not be verified: {exc}") from exc
    cred_id = bytes_to_base64url(verified.credential_id)
    if db.query(WebAuthnCredential).filter(WebAuthnCredential.credential_id == cred_id).first():
        raise ApiError(409, "CREDENTIAL_EXISTS", "This passkey is already registered")
    transports = credential.get("response", {}).get("transports") or []
    cred = WebAuthnCredential(
        user_id=user.id, credential_id=cred_id, public_key=verified.credential_public_key,
        sign_count=verified.sign_count, aaguid=str(verified.aaguid) if verified.aaguid else None,
        device_type=getattr(verified.credential_device_type, "value", str(verified.credential_device_type)),
        backed_up=bool(verified.credential_backed_up), transports=",".join(map(str, transports))[:200] or None,
        name=(name or "Passkey").strip()[:100] or "Passkey")
    db.add(cred)
    db.commit()
    audit(db, user.id, "PASSKEY_REGISTERED", "user", user.id, {"credential": cred.id, "name": cred.name}, ip,
          org_id=org_id)
    log_security_event(db, "PASSKEY_REGISTERED", f"Passkey '{cred.name}' registered for {user.email}", "WARNING",
                       user_id=user.id, ip_address=ip, org_id=org_id)
    _security_notice(db, user, "Passkey added", f"A passkey named '{cred.name}' was added to your account.")
    return cred


def authentication_options(db: Session, kind: str, ip: str, user: Optional[User] = None,
                           parent_id: Optional[str] = None) -> dict[str, Any]:
    """Options for a passkey assertion. With no user: discoverable (usernameless) sign-in."""
    challenge = secrets.token_bytes(32)
    creds = active_credentials(db, user.id) if user else []
    opts = generate_authentication_options(rp_id=settings.webauthn_rp_id, challenge=challenge, timeout=300_000,
                                           allow_credentials=_descriptors(creds),
                                           user_verification=UserVerificationRequirement.REQUIRED)
    row = _new_challenge(db, kind, user_id=user.id if user else None, challenge=bytes_to_base64url(challenge),
                         data={"parent": parent_id} if parent_id else None, ip=ip)
    return {"challenge_id": row.id, "options": json.loads(options_to_json(opts))}


def verify_assertion(db: Session, challenge_id: str, kind: str, credential: dict, ip: str,
                     expected_user: Optional[User] = None) -> tuple[User, WebAuthnCredential]:
    row = _load_challenge(db, challenge_id, kind, expected_user.id if expected_user else None)
    _consume(db, row)
    raw_id = str(credential.get("rawId") or credential.get("id") or "")[:512]
    cred = (db.query(WebAuthnCredential).filter(WebAuthnCredential.credential_id == raw_id).first())
    if cred is None or cred.revoked_at is not None:
        log_security_event(db, "PASSKEY_LOGIN_FAILED", "Unknown or revoked passkey presented", "WARNING",
                           user_id=cred.user_id if cred else None, ip_address=ip,
                           metadata={"revoked": bool(cred and cred.revoked_at)})
        raise ApiError(401, "PASSKEY_NOT_RECOGNISED", "This passkey is not registered or has been removed")
    user = db.query(User).filter(User.id == cred.user_id).first()
    if user is None or not user.is_active or (expected_user and user.id != expected_user.id):
        raise ApiError(401, "PASSKEY_NOT_RECOGNISED", "This passkey cannot be used for this account")
    handle = (credential.get("response") or {}).get("userHandle")
    if handle and base64url_to_bytes(handle) != user_handle(user):
        raise ApiError(401, "PASSKEY_NOT_RECOGNISED", "Passkey user handle does not match")
    try:
        verified = verify_authentication_response(
            credential=credential, expected_challenge=base64url_to_bytes(row.challenge),
            expected_rp_id=settings.webauthn_rp_id, expected_origin=_rp_origins(),
            credential_public_key=cred.public_key, credential_current_sign_count=cred.sign_count,
            require_user_verification=True)
    except InvalidAuthenticationResponse as exc:
        log_security_event(db, "PASSKEY_LOGIN_FAILED", f"{user.email}: {str(exc)[:200]}", "WARNING",
                           user_id=user.id, ip_address=ip)
        recent = (db.query(AuthChallenge).filter(AuthChallenge.kind == kind, AuthChallenge.user_id == user.id,
                                                 AuthChallenge.created_at >= datetime.utcnow() - timedelta(minutes=15))
                  .count())
        if recent >= 3:
            alert_service.raise_for_user(db, user.id, "PASSKEY_FAILURES", "HIGH",
                                         "Repeated invalid passkey assertions", window_minutes=30)
        raise ApiError(401, "PASSKEY_INVALID", "The passkey response could not be verified") from exc
    cred.sign_count = verified.new_sign_count
    cred.last_used_at = datetime.utcnow()
    db.commit()
    return user, cred


def revoke_credential(db: Session, user: User, credential_id: str, ip: str, org_id: Optional[str],
                      by_admin: Optional[str] = None) -> WebAuthnCredential:
    cred = (db.query(WebAuthnCredential)
            .filter(WebAuthnCredential.id == credential_id, WebAuthnCredential.user_id == user.id).first())
    if cred is None or cred.revoked_at is not None:
        raise ApiError(404, "NOT_FOUND", "Passkey not found")
    cred.revoked_at = datetime.utcnow()
    db.commit()
    audit(db, by_admin or user.id, "PASSKEY_REVOKED", "user", user.id, {"credential": cred.id, "name": cred.name}, ip,
          org_id=org_id)
    log_security_event(db, "PASSKEY_REVOKED", f"Passkey '{cred.name}' removed from {user.email}", "WARNING",
                       user_id=user.id, ip_address=ip, org_id=org_id)
    _security_notice(db, user, "Passkey removed", f"The passkey '{cred.name}' was removed from your account.")
    return cred


# ── TOTP ────────────────────────────────────────────────────────────────────

def _totp(factor: MfaFactor) -> pyotp.TOTP:
    return pyotp.TOTP(crypto.unseal(factor.secret_encrypted).decode())


def active_totp(db: Session, user_id: str) -> Optional[MfaFactor]:
    return (db.query(MfaFactor)
            .filter(MfaFactor.user_id == user_id, MfaFactor.kind == FactorKind.TOTP, MfaFactor.status == "ACTIVE")
            .first())


def begin_totp(db: Session, user: User, label: str = "Authenticator app") -> tuple[MfaFactor, str, str]:
    for old in (db.query(MfaFactor).filter(MfaFactor.user_id == user.id, MfaFactor.kind == FactorKind.TOTP,
                                           MfaFactor.status == "PENDING")):
        old.status = "REVOKED"
        old.revoked_at = datetime.utcnow()
    secret = pyotp.random_base32(length=32)
    factor = MfaFactor(user_id=user.id, kind=FactorKind.TOTP, label=(label or "Authenticator app")[:100],
                       secret_encrypted=crypto.seal(secret.encode()), status="PENDING")
    db.add(factor)
    db.commit()
    uri = pyotp.TOTP(secret).provisioning_uri(name=user.email, issuer_name=TOTP_ISSUER)
    return factor, secret, uri


def _match_totp(factor: MfaFactor, code: str, now: Optional[float] = None) -> Optional[int]:
    """Return the matched time-step, or None. Accepts the previous/current/next step (clock skew)."""
    code = (code or "").strip().replace(" ", "")
    if not code.isdigit() or len(code) != 6:
        return None
    totp = _totp(factor)
    now = time.time() if now is None else now
    step = int(now // totp.interval)
    for s in (step, step - 1, step + 1):
        if crypto.constant_time_equals(totp.generate_otp(s), code):
            return s
    return None


def verify_totp(db: Session, factor: MfaFactor, code: str) -> bool:
    step = _match_totp(factor, code)
    if step is None:
        return False
    if factor.last_used_step is not None and step <= factor.last_used_step:
        return False          # replay of an already-accepted code
    n = (db.query(MfaFactor)
         .filter(MfaFactor.id == factor.id,
                 (MfaFactor.last_used_step.is_(None)) | (MfaFactor.last_used_step < step))
         .update({"last_used_step": step, "last_used_at": datetime.utcnow()}, synchronize_session=False))
    db.commit()
    return n == 1


def confirm_totp(db: Session, user: User, factor_id: str, code: str, ip: str, org_id: Optional[str]) -> tuple[MfaFactor, list[str]]:
    factor = (db.query(MfaFactor)
              .filter(MfaFactor.id == factor_id, MfaFactor.user_id == user.id, MfaFactor.status == "PENDING").first())
    if factor is None:
        raise ApiError(404, "NOT_FOUND", "Authenticator setup not found. Start again.")
    if not verify_totp(db, factor, code):
        raise ApiError(400, "INVALID_CODE", "That code is not correct. Check the time on your device and retry.")
    for old in active_factors(db, user.id, FactorKind.TOTP):
        if old.id != factor.id:
            old.status, old.revoked_at = "REVOKED", datetime.utcnow()
    factor.status = "ACTIVE"
    factor.confirmed_at = datetime.utcnow()
    db.commit()
    codes = []
    if not unused_recovery_codes(db, user.id):
        codes = regenerate_recovery_codes(db, user, ip, org_id, notify=False)
    audit(db, user.id, "TOTP_ENABLED", "user", user.id, {"factor": factor.id}, ip, org_id=org_id)
    log_security_event(db, "TOTP_ENABLED", f"Authenticator app enabled for {user.email}", "WARNING",
                       user_id=user.id, ip_address=ip, org_id=org_id)
    _security_notice(db, user, "Authenticator app enabled", "Two-step sign-in with an authenticator app is now on.")
    return factor, codes


def active_factors(db: Session, user_id: str, kind: Optional[str] = None) -> list[MfaFactor]:
    q = db.query(MfaFactor).filter(MfaFactor.user_id == user_id, MfaFactor.status == "ACTIVE")
    if kind:
        q = q.filter(MfaFactor.kind == kind)
    return q.all()


def remove_factor(db: Session, user: User, factor_id: str, ip: str, org_id: Optional[str]) -> MfaFactor:
    f = db.query(MfaFactor).filter(MfaFactor.id == factor_id, MfaFactor.user_id == user.id).first()
    if f is None or f.status == "REVOKED":
        raise ApiError(404, "NOT_FOUND", "Factor not found")
    f.status, f.revoked_at = "REVOKED", datetime.utcnow()
    db.commit()
    audit(db, user.id, "MFA_FACTOR_REMOVED", "user", user.id, {"factor": f.id, "kind": f.kind}, ip, org_id=org_id)
    log_security_event(db, "MFA_FACTOR_REMOVED", f"{f.kind} factor removed for {user.email}", "WARNING",
                       user_id=user.id, ip_address=ip, org_id=org_id)
    _security_notice(db, user, "Sign-in factor removed", f"Your {f.label} was removed from your account.")
    return f


# ── recovery codes ──────────────────────────────────────────────────────────

def _format_code(raw: str) -> str:
    return f"{raw[:4]}-{raw[4:8]}-{raw[8:]}"


def _normalise_code(code: str) -> str:
    return (code or "").strip().lower().replace("-", "").replace(" ", "")


def unused_recovery_codes(db: Session, user_id: str) -> int:
    return db.query(RecoveryCode).filter(RecoveryCode.user_id == user_id, RecoveryCode.used_at.is_(None)).count()


def regenerate_recovery_codes(db: Session, user: User, ip: str, org_id: Optional[str], notify: bool = True) -> list[str]:
    db.query(RecoveryCode).filter(RecoveryCode.user_id == user.id, RecoveryCode.used_at.is_(None)).update(
        {"used_at": datetime.utcnow()}, synchronize_session=False)
    batch = secrets.token_hex(8)
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
    codes = []
    for _ in range(RECOVERY_CODE_COUNT):
        raw = "".join(secrets.choice(alphabet) for _ in range(10))
        codes.append(_format_code(raw))
        db.add(RecoveryCode(user_id=user.id, batch_id=batch, code_hash=crypto.keyed_hash(raw, "recovery")))
    db.commit()
    audit(db, user.id, "RECOVERY_CODES_GENERATED", "user", user.id, {"count": len(codes)}, ip, org_id=org_id)
    if notify:
        _security_notice(db, user, "New recovery codes", "Your previous recovery codes no longer work.")
    return codes


def consume_recovery_code(db: Session, user: User, code: str) -> bool:
    h = crypto.keyed_hash(_normalise_code(code), "recovery")
    n = (db.query(RecoveryCode)
         .filter(RecoveryCode.user_id == user.id, RecoveryCode.code_hash == h, RecoveryCode.used_at.is_(None))
         .update({"used_at": datetime.utcnow()}, synchronize_session=False))
    db.commit()
    if n == 1:
        _security_notice(db, user, "Recovery code used",
                         f"A recovery code was used to sign in. {unused_recovery_codes(db, user.id)} codes remain.")
    return n == 1


# ── email one-time codes ────────────────────────────────────────────────────

def send_email_otp(db: Session, user: User, parent: AuthChallenge, ip: str) -> None:
    if not email_service.available():
        raise ApiError(503, "EMAIL_UNAVAILABLE", "Email codes are not available: no mail server is configured")
    code = crypto.numeric_code(6)
    db.query(AuthChallenge).filter(AuthChallenge.kind == ChallengeKind.EMAIL_OTP, AuthChallenge.user_id == user.id,
                                   AuthChallenge.consumed_at.is_(None)).update(
        {"consumed_at": datetime.utcnow()}, synchronize_session=False)
    _new_challenge(db, ChallengeKind.EMAIL_OTP, user_id=user.id, token=f"{parent.id}:{code}",
                   data={"parent": parent.id}, ttl=EMAIL_OTP_TTL, ip=ip)
    try:
        email_service.send(user.email, "Your FaceVerification sign-in code",
                           f"Your sign-in code is {code}. It expires in 10 minutes.\n\n"
                           "If you did not try to sign in, change your password.")
    except email_service.EmailSendError as exc:
        raise ApiError(502, "EMAIL_FAILED", "The email could not be sent. Use another method.") from exc


def verify_email_otp(db: Session, user: User, parent: AuthChallenge, code: str) -> bool:
    code = (code or "").strip()
    if not code.isdigit() or len(code) != 6:
        return False
    row = (db.query(AuthChallenge)
           .filter(AuthChallenge.kind == ChallengeKind.EMAIL_OTP, AuthChallenge.user_id == user.id,
                   AuthChallenge.token_hash == crypto.keyed_hash(f"{parent.id}:{code}", "authchallenge"),
                   AuthChallenge.consumed_at.is_(None), AuthChallenge.expires_at >= datetime.utcnow()).first())
    if row is None:
        return False
    n = (db.query(AuthChallenge).filter(AuthChallenge.id == row.id, AuthChallenge.consumed_at.is_(None))
         .update({"consumed_at": datetime.utcnow()}, synchronize_session=False))
    db.commit()
    return n == 1


# ── sign-in orchestration ───────────────────────────────────────────────────

def strong_methods_available(db: Session, user: User) -> list[str]:
    methods = []
    if active_credentials(db, user.id):
        methods.append("webauthn")
    if active_totp(db, user.id):
        methods.append("totp")
    if unused_recovery_codes(db, user.id):
        methods.append("recovery_code")
    return methods


def mfa_required_by_policy(db: Session, user: User) -> bool:
    for org, m in active_orgs(db, user.id):
        if m.role == RoleEnum.ADMIN and get_active_policy(db, org.id).mfa_required_for_admins:
            return True
    return False


@dataclass
class LoginStep:
    done: bool
    pending_token: Optional[str] = None
    kind: Optional[str] = None
    methods: Optional[list[str]] = None
    enrol_required: bool = False
    org_id: Optional[str] = None


def after_password(db: Session, user: User, ip: str) -> LoginStep:
    """Decide whether a password sign-in needs a second factor."""
    has_second = bool(active_totp(db, user.id) or active_credentials(db, user.id))
    required = has_second or mfa_required_by_policy(db, user)
    if not required:
        return LoginStep(done=True)
    methods = strong_methods_available(db, user)
    if email_service.available():
        methods.append("email_otp")
    token = crypto.token(32)
    _new_challenge(db, ChallengeKind.MFA_PENDING, user_id=user.id, token=token,
                   data={"amr": ["pwd"], "enrol_required": not has_second}, ttl=PENDING_TTL, ip=ip)
    return LoginStep(done=False, pending_token=token, kind="mfa_required", methods=methods,
                     enrol_required=not has_second)


def after_passkey(db: Session, user: User, ip: str) -> LoginStep:
    """A passkey assertion is complete; organisation policy may still ask for a live face check."""
    org_id = pick_org(db, user)
    if org_id is None:
        raise ApiError(403, "NO_ORGANISATION", "Your account is not a member of any active organisation")
    if get_active_policy(db, org_id).passkey_login_requires_face:
        from backend.services.enrolment_service import get_enrolment
        enr = get_enrolment(db, org_id, user.id)
        if enr is not None and enr.is_usable:
            token = crypto.token(32)
            _new_challenge(db, ChallengeKind.FACE_PENDING, user_id=user.id, org_id=org_id, token=token,
                           data={"amr": ["webauthn"], "sessions": []}, ttl=PENDING_TTL, ip=ip)
            return LoginStep(done=False, pending_token=token, kind="face_required", org_id=org_id)
    return LoginStep(done=True, org_id=org_id)


def complete_mfa(db: Session, pending: AuthChallenge, user: User, method: str, ip: str, code: str = "",
                 challenge_id: str = "", credential: Optional[dict] = None) -> list[str]:
    """Verify the second factor of a pending password sign-in. Returns the session `amr`."""
    data = _data(pending)
    ok = False
    if method == "totp":
        f = active_totp(db, user.id)
        ok = f is not None and verify_totp(db, f, code)
    elif method == "recovery_code":
        ok = consume_recovery_code(db, user, code)
    elif method == "email_otp":
        ok = verify_email_otp(db, user, pending, code)
    elif method == "webauthn":
        try:
            verify_assertion(db, challenge_id, ChallengeKind.WEBAUTHN_STEP_UP, credential or {}, ip, expected_user=user)
            ok = True
        except ApiError:
            ok = False
    else:
        raise ApiError(422, "UNKNOWN_METHOD", "Unknown verification method")
    if not ok:
        _failed_attempt(db, pending, user, method, ip)
        raise ApiError(401, "INVALID_CODE", "Verification failed. Check the code and try again.")
    _consume(db, pending)
    amr = list(data.get("amr", ["pwd"])) + [{"recovery_code": "rc", "email_otp": "otp"}.get(method, method)]
    log_security_event(db, "MFA_SUCCESS", f"{user.email} completed {method}", "INFO", user_id=user.id, ip_address=ip)
    return amr


# ── step-up ─────────────────────────────────────────────────────────────────

def step_up(db: Session, user: User, method: str, ip: str, password: str = "", code: str = "",
            challenge_id: str = "", credential: Optional[dict] = None) -> list[str]:
    """Re-authenticate for sensitive actions. Password is accepted only when no stronger factor exists."""
    strong = strong_methods_available(db, user)
    ok = False
    if method == "password":
        if [m for m in strong if m != "recovery_code"]:
            raise ApiError(400, "STRONGER_FACTOR_REQUIRED", "Use your passkey or authenticator app to confirm")
        ok = verify_password(password, user.hashed_password)
    elif method == "totp":
        f = active_totp(db, user.id)
        ok = f is not None and verify_totp(db, f, code)
    elif method == "recovery_code":
        ok = consume_recovery_code(db, user, code)
    elif method == "webauthn":
        verify_assertion(db, challenge_id, ChallengeKind.WEBAUTHN_STEP_UP, credential or {}, ip, expected_user=user)
        ok = True
    else:
        raise ApiError(422, "UNKNOWN_METHOD", "Unknown verification method")
    if not ok:
        log_security_event(db, "STEP_UP_FAILED", f"Step-up ({method}) failed for {user.email}", "WARNING",
                           user_id=user.id, ip_address=ip)
        raise ApiError(401, "STEP_UP_FAILED", "Confirmation failed")
    log_security_event(db, "STEP_UP", f"{user.email} confirmed identity ({method})", "INFO", user_id=user.id,
                       ip_address=ip)
    return [{"password": "pwd", "recovery_code": "rc"}.get(method, method)]


# ── account security overview / admin reset ─────────────────────────────────

def security_overview(db: Session, user: User) -> dict[str, Any]:
    return {
        "passkeys": [{"id": c.id, "name": c.name, "created_at": c.created_at, "last_used_at": c.last_used_at,
                      "device_type": c.device_type, "backed_up": c.backed_up,
                      "transports": c.transports.split(",") if c.transports else []}
                     for c in active_credentials(db, user.id)],
        "factors": [{"id": f.id, "kind": f.kind, "label": f.label, "created_at": f.created_at,
                     "confirmed_at": f.confirmed_at, "last_used_at": f.last_used_at}
                    for f in active_factors(db, user.id)],
        "recovery_codes_remaining": unused_recovery_codes(db, user.id),
        "email_otp_available": email_service.available(),
        "mfa_required_by_policy": mfa_required_by_policy(db, user),
        "second_factor_enabled": bool(active_totp(db, user.id) or active_credentials(db, user.id)),
    }


def admin_reset_factors(db: Session, admin_id: str, user: User, ip: str, org_id: str) -> dict[str, int]:
    now = datetime.utcnow()
    p = (db.query(WebAuthnCredential).filter(WebAuthnCredential.user_id == user.id, WebAuthnCredential.revoked_at.is_(None))
         .update({"revoked_at": now}, synchronize_session=False))
    f = (db.query(MfaFactor).filter(MfaFactor.user_id == user.id, MfaFactor.status != "REVOKED")
         .update({"status": "REVOKED", "revoked_at": now}, synchronize_session=False))
    r = (db.query(RecoveryCode).filter(RecoveryCode.user_id == user.id, RecoveryCode.used_at.is_(None))
         .update({"used_at": now}, synchronize_session=False))
    db.commit()
    audit(db, admin_id, "MFA_RESET_BY_ADMIN", "user", user.id, {"passkeys": p, "factors": f, "recovery_codes": r},
          ip, org_id=org_id)
    log_security_event(db, "MFA_RESET_BY_ADMIN", f"Sign-in factors of {user.email} reset by an administrator", "HIGH",
                       user_id=admin_id, ip_address=ip, metadata={"target_user_id": user.id}, org_id=org_id)
    _security_notice(db, user, "Sign-in factors reset", "An administrator removed your passkeys and authenticator "
                     "apps. Set them up again from Account security.")
    return {"passkeys": p, "factors": f, "recovery_codes": r}


def _security_notice(db: Session, user: User, title: str, body: str) -> None:
    for m in db.query(Membership).filter(Membership.user_id == user.id, Membership.status == MembershipStatus.ACTIVE):
        notification_service.notify(db, m.organization_id, user.id, "ACCOUNT_SECURITY", title, body,
                                    link="/account/security", severity="WARNING")
    db.commit()


def expire_challenges(db: Session, now: datetime, _policies: dict) -> dict[str, int]:
    n = (db.query(AuthChallenge).filter(AuthChallenge.expires_at < now - timedelta(hours=1))
         .delete(synchronize_session=False))
    return {"auth_challenges_deleted": n}


# ── live face check after a passkey (or for face step-up) ───────────────────

def face_actor(db: Session, token: str):
    """Capture principal for a pending face check. It owns only the sessions it creates."""
    from backend.core.deps import Actor
    row = load_pending(db, token, ChallengeKind.FACE_PENDING)
    return Actor(kind="LOGIN", org_id=row.organization_id, ref=row.id, label="face check")


def start_face_session(db: Session, token: str, ip: str):
    from backend.models.verification import SessionPurpose
    from backend.services import capture_service
    row = load_pending(db, token, ChallengeKind.FACE_PENDING)
    data = _data(row)
    if len(data.get("sessions", [])) >= MAX_FACE_SESSIONS:
        _consume(db, row)
        raise ApiError(429, "TOO_MANY_ATTEMPTS", "Too many face-check attempts. Start again.")
    user = db.query(User).filter(User.id == row.user_id).first()
    actor = face_actor(db, token)
    created = capture_service.create_session(db, actor, SessionPurpose.VERIFY, user, ip, channel="LOGIN",
                                             created_by_user_id=user.id)
    data.setdefault("sessions", []).append(created.session.id)
    _set_data(db, row, data)
    view = capture_service.session_view(db, created.session, actor, state=created.state)
    view["frame_token"] = created.frame_token
    return view


def face_result(db: Session, token: str, ip: str) -> tuple[User, AuthChallenge, list[str]]:
    """Consume a FACE_PENDING challenge whose latest face session VERIFIED the same user."""
    from backend.models.verification import VerificationSession
    row = load_pending(db, token, ChallengeKind.FACE_PENDING)
    data = _data(row)
    sessions = data.get("sessions", [])
    if not sessions:
        raise ApiError(409, "FACE_CHECK_REQUIRED", "Complete the face check first")
    s = db.query(VerificationSession).filter(VerificationSession.id == sessions[-1]).first()
    if (s is None or s.owner_kind != "LOGIN" or s.owner_ref != row.id or s.target_user_id != row.user_id
            or s.is_active):
        raise ApiError(409, "FACE_CHECK_REQUIRED", "Complete the face check first")
    user = db.query(User).filter(User.id == row.user_id).first()
    if s.final_outcome != "VERIFIED":
        _failed_attempt(db, row, user, f"face check ({s.final_outcome})", ip)
        raise ApiError(401, "FACE_CHECK_FAILED", f"Face check did not pass ({s.final_outcome}). Try again.")
    _consume(db, row)
    log_security_event(db, "FACE_CHECK_PASSED", f"Live face check passed for {user.email}", "INFO", user_id=user.id,
                       ip_address=ip, org_id=row.organization_id)
    return user, row, list(data.get("amr", [])) + ["face"]


def begin_face_step_up(db: Session, user: User, org_id: str, ip: str) -> str:
    from backend.services.enrolment_service import get_enrolment
    enr = get_enrolment(db, org_id, user.id)
    if enr is None or not enr.is_usable:
        raise ApiError(409, "NOT_ENROLLED", "You need an active biometric enrolment in this organisation for a face check")
    token = crypto.token(32)
    _new_challenge(db, ChallengeKind.FACE_PENDING, user_id=user.id, org_id=org_id, token=token,
                   data={"amr": [], "sessions": [], "purpose": "step_up"}, ttl=PENDING_TTL, ip=ip)
    return token
