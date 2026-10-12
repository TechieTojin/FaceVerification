"""Authentication service — v2.1: refresh tokens, lockout tracking."""
from __future__ import annotations
import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models.user import User, RoleEnum
from backend.models.auth import RefreshToken, LoginAttempt

log = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Password helpers ────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── Access tokens ───────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    payload["exp"] = expire
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except JWTError:
        return None


# ── Refresh tokens ──────────────────────────────────────────────────────────

def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def create_refresh_token(db: Session, user_id: str) -> str:
    """Create a refresh token, persist its hash, return the raw value."""
    raw = secrets.token_urlsafe(64)
    token_hash = _hash_token(raw)
    expires_at = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
    db.add(RefreshToken(token_hash=token_hash, user_id=user_id, expires_at=expires_at))
    db.commit()
    return raw


def validate_and_rotate_refresh_token(db: Session, raw: str) -> Optional[tuple[str, User]]:
    """
    Validate a refresh token, revoke it, issue a replacement.
    Returns (new_raw_refresh_token, user) or None if invalid/revoked/expired.

    Detects token reuse: if an already-revoked token is presented,
    all tokens for that user are revoked (compromise assumed).
    """
    token_hash = _hash_token(raw)
    rt: Optional[RefreshToken] = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash,
    ).first()

    if not rt:
        return None

    if rt.revoked:
        # Token reuse detected — revoke all tokens for this user
        log.warning("Refresh token reuse detected for user %s — revoking all tokens", rt.user_id)
        _revoke_all(db, rt.user_id)
        return None

    if rt.expires_at < datetime.utcnow():
        rt.revoked = True
        rt.revoked_at = datetime.utcnow()
        db.commit()
        return None

    user = get_user_by_id(db, rt.user_id)
    if not user or not user.is_active:
        rt.revoked = True
        rt.revoked_at = datetime.utcnow()
        db.commit()
        return None

    # Rotate: revoke old, issue new
    new_raw = secrets.token_urlsafe(64)
    new_hash = _hash_token(new_raw)
    rt.revoked = True
    rt.revoked_at = datetime.utcnow()
    rt.replaced_by = new_hash

    new_expires = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
    db.add(RefreshToken(token_hash=new_hash, user_id=user.id, expires_at=new_expires))
    db.commit()

    return new_raw, user


def revoke_refresh_token(db: Session, raw: str) -> bool:
    token_hash = _hash_token(raw)
    rt = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
    if rt and not rt.revoked:
        rt.revoked = True
        rt.revoked_at = datetime.utcnow()
        db.commit()
        return True
    return False


def _revoke_all(db: Session, user_id: str) -> int:
    now = datetime.utcnow()
    count = db.query(RefreshToken).filter(
        RefreshToken.user_id == user_id,
        RefreshToken.revoked == False,
    ).update({"revoked": True, "revoked_at": now})
    db.commit()
    return count


def revoke_all_refresh_tokens(db: Session, user_id: str) -> int:
    return _revoke_all(db, user_id)


# ── Rate limiting / lockout ──────────────────────────────────────────────────

def record_login_attempt(db: Session, email: str, ip: str, success: bool) -> None:
    db.add(LoginAttempt(
        identifier=email.lower(),
        identifier_type="email",
        success=success,
        ip_address=ip,
    ))
    db.add(LoginAttempt(
        identifier=ip,
        identifier_type="ip",
        success=success,
        ip_address=ip,
    ))
    db.commit()


def check_rate_limit(db: Session, email: str, ip: str) -> tuple[bool, str, int]:
    """
    Returns (allowed, reason, retry_after_seconds).
    allowed=False → reject login with 429.
    """
    now = datetime.utcnow()
    window_start = now - timedelta(seconds=settings.login_rate_limit_window_seconds)
    lockout_window = now - timedelta(seconds=settings.login_lockout_duration_seconds)

    # 1. Account lockout — N consecutive failures with no success between them
    recent = (
        db.query(LoginAttempt)
        .filter(
            LoginAttempt.identifier == email.lower(),
            LoginAttempt.identifier_type == "email",
            LoginAttempt.attempted_at >= lockout_window,
        )
        .order_by(LoginAttempt.attempted_at.desc())
        .limit(settings.login_lockout_threshold + 1)
        .all()
    )
    consecutive_failures = 0
    for attempt in recent:
        if attempt.success:
            break
        consecutive_failures += 1
    if consecutive_failures >= settings.login_lockout_threshold:
        oldest = recent[consecutive_failures - 1]
        unlock_at = oldest.attempted_at + timedelta(seconds=settings.login_lockout_duration_seconds)
        retry_after = max(0, int((unlock_at - now).total_seconds()))
        return False, "account_locked", retry_after

    # 2. Per-IP rate limit
    ip_count = (
        db.query(LoginAttempt)
        .filter(
            LoginAttempt.identifier == ip,
            LoginAttempt.identifier_type == "ip",
            LoginAttempt.attempted_at >= window_start,
        )
        .count()
    )
    if ip_count >= settings.login_rate_limit_per_ip:
        return False, "ip_rate_limited", settings.login_rate_limit_window_seconds

    # 3. Per-account rate limit (all attempts, not just failures)
    acct_count = (
        db.query(LoginAttempt)
        .filter(
            LoginAttempt.identifier == email.lower(),
            LoginAttempt.identifier_type == "email",
            LoginAttempt.attempted_at >= window_start,
        )
        .count()
    )
    if acct_count >= settings.login_rate_limit_per_account:
        return False, "account_rate_limited", settings.login_rate_limit_window_seconds

    return True, "", 0


# ── User lookup ─────────────────────────────────────────────────────────────

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, email: str, full_name: str, password: str,
                role: RoleEnum = RoleEnum.OPERATOR) -> User:
    user = User(
        email=email,
        full_name=full_name,
        hashed_password=hash_password(password),
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def seed_admin(db: Session) -> None:
    """Create default admin if DB has no users. Credentials from env only."""
    if not settings.initial_admin_password:
        log.warning(
            "INITIAL_ADMIN_PASSWORD not set in .env — skipping admin seed. "
            "Set it and restart to create the first admin account."
        )
        return
    if db.query(User).count() == 0:
        admin = create_user(
            db,
            email=settings.initial_admin_email,
            full_name=settings.initial_admin_name,
            password=settings.initial_admin_password,
            role=RoleEnum.ADMIN,
        )
        log.warning("Default admin created: %s — CHANGE THIS PASSWORD!", admin.email)
