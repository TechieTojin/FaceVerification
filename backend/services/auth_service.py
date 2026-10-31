"""Authentication primitives: passwords, JWT access tokens, rotating refresh tokens, rate limiting.

Access-token claims: `sub` (user), `org` (active organisation), `amr` (methods
used at sign-in: pwd, webauthn, totp, otp, rc, face), `auth_time`, and `su`
(time of the last step-up). The organisation claim is re-validated against an
active membership on every request.
"""
from __future__ import annotations

import calendar
import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models.auth import LoginAttempt, RefreshToken
from backend.models.organization import Membership, MembershipStatus, Organization, OrgStatus
from backend.models.user import RoleEnum, User

log = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=settings.bcrypt_rounds)
_DUMMY_HASH = pwd_context.hash("timing-equaliser-not-a-real-password")
MAX_PASSWORD_BYTES = 72  # bcrypt only uses the first 72 bytes


class PasswordPolicyError(ValueError):
    pass


def validate_password_strength(password: str) -> None:
    if len(password) < 10:
        raise PasswordPolicyError("Password must be at least 10 characters")
    if len(password.encode()) > MAX_PASSWORD_BYTES:
        raise PasswordPolicyError("Password must be at most 72 bytes")
    classes = sum([any(c.islower() for c in password), any(c.isupper() for c in password),
                   any(c.isdigit() for c in password), any(not c.isalnum() for c in password)])
    if classes < 3:
        raise PasswordPolicyError("Password must contain at least three of: lowercase, uppercase, digit, symbol")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    if len(plain.encode()) > MAX_PASSWORD_BYTES:
        return False
    try:
        return pwd_context.verify(plain, hashed)
    except ValueError:
        return False


# ── Access tokens ───────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    payload["iat"] = datetime.utcnow()
    payload["typ"] = "access"
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def session_token(user: User, org_id: str, amr: list[str], auth_time: int, step_up_at: Optional[int] = None,
                  step_up_amr: Optional[list[str]] = None) -> str:
    claims = {"sub": user.id, "org": org_id, "amr": amr, "auth_time": auth_time}
    if step_up_at:
        claims["su"] = step_up_at
        claims["su_amr"] = step_up_amr or []
    return create_access_token(claims)


def active_orgs(db: Session, user_id: str) -> list[tuple[Organization, Membership]]:
    return (db.query(Organization, Membership)
            .join(Membership, Membership.organization_id == Organization.id)
            .filter(Membership.user_id == user_id, Membership.status == MembershipStatus.ACTIVE,
                    Organization.status == OrgStatus.ACTIVE)
            .order_by(Organization.name).all())


def pick_org(db: Session, user: User, preferred: Optional[str] = None) -> Optional[str]:
    """The organisation a new sign-in lands in: preferred → last used → first active membership."""
    orgs = [o.id for o, _ in active_orgs(db, user.id)]
    for candidate in (preferred, user.last_org_id):
        if candidate and candidate in orgs:
            return candidate
    return orgs[0] if orgs else None


def decode_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except JWTError:
        return None
    if payload.get("typ", "access") != "access":
        return None
    return payload


# ── Refresh tokens ──────────────────────────────────────────────────────────

def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def create_refresh_token(db: Session, user_id: str, user_agent: str = "", ip: str = "",
                         family_id: Optional[str] = None, org_id: Optional[str] = None,
                         amr: Optional[list[str]] = None) -> str:
    raw = secrets.token_urlsafe(64)
    db.add(RefreshToken(
        token_hash=_hash_token(raw), user_id=user_id, family_id=family_id or str(uuid.uuid4()),
        organization_id=org_id, amr=",".join(amr or ["pwd"]),
        expires_at=datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days),
        user_agent=user_agent[:200] or None, ip_address=ip or None,
    ))
    db.commit()
    return raw


class RefreshResult:
    def __init__(self, raw: Optional[str] = None, user: Optional[User] = None, reuse_detected: bool = False,
                 reuse_user_id: Optional[str] = None, org_id: Optional[str] = None,
                 amr: Optional[list[str]] = None, auth_time: Optional[int] = None):
        self.raw, self.user, self.reuse_detected, self.reuse_user_id = raw, user, reuse_detected, reuse_user_id
        self.org_id, self.amr, self.auth_time = org_id, amr or ["pwd"], auth_time

    @property
    def ok(self) -> bool:
        return self.raw is not None


def validate_and_rotate_refresh_token(db: Session, raw: str, user_agent: str = "", ip: str = "") -> RefreshResult:
    """Validate, revoke and replace a refresh token.

    Presenting an already-rotated token means the token was copied: every
    token for that user is revoked and `reuse_detected` is set.
    """
    rt = db.query(RefreshToken).filter(RefreshToken.token_hash == _hash_token(raw)).first()
    if not rt:
        return RefreshResult()
    now = datetime.utcnow()
    if rt.revoked:
        log.warning("Refresh token reuse detected for user %s — revoking all tokens", rt.user_id)
        _revoke_all(db, rt.user_id)
        return RefreshResult(reuse_detected=True, reuse_user_id=rt.user_id)
    if rt.expires_at < now:
        rt.revoked, rt.revoked_at = True, now
        db.commit()
        return RefreshResult()
    user = get_user_by_id(db, rt.user_id)
    if not user or not user.is_active:
        rt.revoked, rt.revoked_at = True, now
        db.commit()
        return RefreshResult()
    org_id = pick_org(db, user, rt.organization_id)
    if org_id is None:
        rt.revoked, rt.revoked_at = True, now
        db.commit()
        return RefreshResult()
    new_raw = secrets.token_urlsafe(64)
    new_hash = _hash_token(new_raw)
    rt.revoked, rt.revoked_at, rt.replaced_by = True, now, new_hash
    db.add(RefreshToken(token_hash=new_hash, user_id=user.id, family_id=rt.family_id,
                        organization_id=org_id, amr=rt.amr,
                        expires_at=now + timedelta(days=settings.refresh_token_expire_days),
                        user_agent=(user_agent or rt.user_agent or "")[:200] or None, ip_address=ip or rt.ip_address))
    db.commit()
    first = (db.query(RefreshToken.created_at).filter(RefreshToken.family_id == rt.family_id)
             .order_by(RefreshToken.created_at).first())
    auth_time = calendar.timegm(first[0].utctimetuple()) if first else None
    return RefreshResult(raw=new_raw, user=user, org_id=org_id, amr=(rt.amr or "pwd").split(","),
                         auth_time=auth_time)


def revoke_refresh_token(db: Session, raw: str) -> bool:
    rt = db.query(RefreshToken).filter(RefreshToken.token_hash == _hash_token(raw)).first()
    if rt and not rt.revoked:
        rt.revoked, rt.revoked_at = True, datetime.utcnow()
        db.commit()
        return True
    return False


def set_family_org(db: Session, raw: Optional[str], org_id: str) -> None:
    """Remember the organisation chosen with /auth/switch-org for the current sign-in."""
    if not raw:
        return
    rt = db.query(RefreshToken).filter(RefreshToken.token_hash == _hash_token(raw)).first()
    if rt and not rt.revoked:
        rt.organization_id = org_id
        db.commit()


def current_family(db: Session, raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    rt = db.query(RefreshToken).filter(RefreshToken.token_hash == _hash_token(raw)).first()
    return rt.family_id if rt else None


def revoke_family(db: Session, user_id: str, family_id: str) -> int:
    n = (db.query(RefreshToken)
         .filter(RefreshToken.user_id == user_id, RefreshToken.family_id == family_id, RefreshToken.revoked == False)  # noqa: E712
         .update({"revoked": True, "revoked_at": datetime.utcnow()}))
    db.commit()
    return n


def _revoke_all(db: Session, user_id: str) -> int:
    n = (db.query(RefreshToken)
         .filter(RefreshToken.user_id == user_id, RefreshToken.revoked == False)  # noqa: E712
         .update({"revoked": True, "revoked_at": datetime.utcnow()}))
    db.commit()
    return n


def revoke_all_refresh_tokens(db: Session, user_id: str) -> int:
    return _revoke_all(db, user_id)


# ── Rate limiting / lockout ──────────────────────────────────────────────────

def record_login_attempt(db: Session, email: str, ip: str, success: bool) -> None:
    db.add(LoginAttempt(identifier=email.lower(), identifier_type="email", success=success, ip_address=ip))
    db.add(LoginAttempt(identifier=ip, identifier_type="ip", success=success, ip_address=ip))
    db.commit()


def check_rate_limit(db: Session, email: str, ip: str) -> tuple[bool, str, int]:
    """Returns (allowed, reason, retry_after_seconds)."""
    now = datetime.utcnow()
    window_start = now - timedelta(seconds=settings.login_rate_limit_window_seconds)
    lockout_window = now - timedelta(seconds=settings.login_lockout_duration_seconds)
    email = email.lower()

    recent = (db.query(LoginAttempt)
              .filter(LoginAttempt.identifier == email, LoginAttempt.identifier_type == "email",
                      LoginAttempt.attempted_at >= lockout_window)
              .order_by(LoginAttempt.attempted_at.desc())
              .limit(settings.login_lockout_threshold + 1).all())
    consecutive = 0
    for attempt in recent:
        if attempt.success:
            break
        consecutive += 1
    if consecutive >= settings.login_lockout_threshold:
        oldest = recent[consecutive - 1]
        unlock_at = oldest.attempted_at + timedelta(seconds=settings.login_lockout_duration_seconds)
        return False, "account_locked", max(1, int((unlock_at - now).total_seconds()))

    ip_count = (db.query(LoginAttempt)
                .filter(LoginAttempt.identifier == ip, LoginAttempt.identifier_type == "ip",
                        LoginAttempt.attempted_at >= window_start).count())
    if ip_count >= settings.login_rate_limit_per_ip:
        return False, "ip_rate_limited", settings.login_rate_limit_window_seconds

    acct_count = (db.query(LoginAttempt)
                  .filter(LoginAttempt.identifier == email, LoginAttempt.identifier_type == "email",
                          LoginAttempt.attempted_at >= window_start).count())
    if acct_count >= settings.login_rate_limit_per_account:
        return False, "account_rate_limited", settings.login_rate_limit_window_seconds
    return True, "", 0


# ── Users ───────────────────────────────────────────────────────────────────

def normalise_email(email: str) -> str:
    return email.strip().lower()


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Constant-work authentication: a bcrypt check runs even for unknown emails."""
    user = db.query(User).filter(User.email == normalise_email(email)).first()
    if not user:
        verify_password(password, _DUMMY_HASH)
        return None
    if not verify_password(password, user.hashed_password):
        return None
    if not user.is_active:
        return None
    return user


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == normalise_email(email)).first()


def create_user(db: Session, email: str, full_name: str, password: str,
                role: RoleEnum = RoleEnum.OPERATOR, commit: bool = True, org_id: Optional[str] = None,
                invited_by: Optional[str] = None, platform_admin: bool = False) -> User:
    user = User(email=normalise_email(email), full_name=full_name.strip(), hashed_password=hash_password(password),
                is_platform_admin=platform_admin, is_active=True, password_changed_at=datetime.utcnow(),
                last_org_id=org_id)
    db.add(user)
    db.flush()
    if org_id:
        db.add(Membership(organization_id=org_id, user_id=user.id, role=role, status=MembershipStatus.ACTIVE,
                          invited_by_user_id=invited_by))
    if commit:
        db.commit()
        db.refresh(user)
    else:
        db.flush()
    return user


def ensure_default_org(db: Session) -> Organization:
    org = db.query(Organization).order_by(Organization.created_at).first()
    if org is None:
        org = Organization(slug="default", name="Default organization", status=OrgStatus.ACTIVE)
        db.add(org)
        db.commit()
    return org


def seed_admin(db: Session) -> None:
    """Create the initial platform admin if the DB has no users. Credentials from env only."""
    org = ensure_default_org(db)
    if db.query(User).count() > 0:
        return
    if not settings.initial_admin_password:
        log.warning("INITIAL_ADMIN_PASSWORD not set — skipping admin seed.")
        return
    admin = create_user(db, settings.initial_admin_email, settings.initial_admin_name,
                        settings.initial_admin_password, RoleEnum.ADMIN, org_id=org.id, platform_admin=True)
    log.warning("Initial admin created: %s — change this password after first login.", admin.email)
