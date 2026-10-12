"""Authentication router — v2.1.

Auth flow:
  Login  → sets fv_access (HttpOnly), fv_refresh (HttpOnly, path=/api/v1/auth),
            fv_csrf (JS-readable); also returns access_token in body for API clients.
  Refresh → validates fv_refresh cookie, rotates token pair.
  Logout  → revokes refresh token, clears all auth cookies.
  /me     → returns current user; used by frontend to restore session on reload.

CSRF: state-changing endpoints protected by X-CSRF-Token double-submit cookie.
      Enforced by CSRFMiddleware in main.py for cookie-authenticated requests.
"""
from __future__ import annotations
import logging
import secrets
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import get_db
from backend.models.user import User
from backend.schemas.auth import (
    LoginRequest, LoginResponse, RefreshResponse,
    PasswordChangeRequest, UserInfo,
)
from backend.services.auth_service import (
    authenticate_user, check_rate_limit, create_access_token,
    create_refresh_token, decode_token, get_user_by_id,
    hash_password, record_login_attempt,
    revoke_refresh_token, revoke_all_refresh_tokens,
    validate_and_rotate_refresh_token, verify_password,
)
from backend.services.audit_service import log_security_event

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

_ACCESS_COOKIE = "fv_access"
_REFRESH_COOKIE = settings.refresh_token_cookie_name
_CSRF_COOKIE = settings.csrf_cookie_name


# ── Cookie helpers ───────────────────────────────────────────────────────────

def _set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
    csrf_token: str,
) -> None:
    secure = settings.cookie_secure
    same_site = settings.cookie_samesite

    response.set_cookie(
        key=_ACCESS_COOKIE,
        value=access_token,
        httponly=True,
        secure=secure,
        samesite=same_site,
        max_age=settings.access_token_expire_minutes * 60,
        path="/",
    )
    response.set_cookie(
        key=_REFRESH_COOKIE,
        value=refresh_token,
        httponly=True,
        secure=secure,
        samesite=same_site,
        max_age=settings.refresh_token_expire_days * 86400,
        path="/api/v1/auth",   # narrowed path — only sent to auth endpoints
    )
    response.set_cookie(
        key=_CSRF_COOKIE,
        value=csrf_token,
        httponly=False,          # JS must read this for the double-submit pattern
        secure=secure,
        samesite=same_site,
        max_age=settings.access_token_expire_minutes * 60,
        path="/",
    )


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(_ACCESS_COOKIE, path="/")
    response.delete_cookie(_REFRESH_COOKIE, path="/api/v1/auth")
    response.delete_cookie(_CSRF_COOKIE, path="/")


# ── Auth dependency ──────────────────────────────────────────────────────────

def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    bearer: Optional[str] = Depends(oauth2_scheme),
) -> User:
    """Accept either HttpOnly cookie or Authorization: Bearer token."""
    token = request.cookies.get(_ACCESS_COOKIE) or bearer

    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise exc
    payload = decode_token(token)
    if not payload:
        raise exc
    user_id = payload.get("sub")
    if not user_id:
        raise exc
    user = get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise exc
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    from backend.models.user import RoleEnum
    if current_user.role != RoleEnum.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def _get_client_ip(request: Request) -> str:
    """Use the direct TCP connection IP — never trust forwarded headers blindly."""
    if request.client:
        return request.client.host
    return "unknown"


def _user_info(user: User) -> UserInfo:
    return UserInfo(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role.value,
        is_active=user.is_active,
        created_at=user.created_at,
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/login", response_model=LoginResponse)
def login(request: Request, body: LoginRequest, response: Response, db: Session = Depends(get_db)):
    ip = _get_client_ip(request)

    # Rate limit / lockout check BEFORE password verification
    # (avoids timing oracle that reveals whether the account exists)
    allowed, reason, retry_after = check_rate_limit(db, body.email, ip)
    if not allowed:
        detail = (
            f"Account temporarily locked. Try again in {retry_after}s."
            if reason == "account_locked"
            else "Too many login attempts. Try again later."
        )
        raise HTTPException(
            status_code=429,
            detail=detail,
            headers={"Retry-After": str(retry_after)},
        )

    user = authenticate_user(db, body.email, body.password)
    success = user is not None

    record_login_attempt(db, body.email, ip, success)

    if not success:
        log_security_event(db, "LOGIN_FAILED", f"Failed login for {body.email}", "WARNING")
        # Always return 401 regardless of whether email exists (anti-enumeration)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token({"sub": user.id, "email": user.email, "role": user.role.value})
    refresh_token = create_refresh_token(db, user.id)
    csrf_token = secrets.token_urlsafe(32)

    _set_auth_cookies(response, access_token, refresh_token, csrf_token)

    user.last_login_at = datetime.utcnow()
    db.commit()
    log_security_event(db, "LOGIN_SUCCESS", f"User {user.email} logged in", "INFO", user_id=user.id)

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
        user=_user_info(user),
    )


@router.post("/refresh", response_model=RefreshResponse)
def refresh(request: Request, response: Response, db: Session = Depends(get_db)):
    """Rotate the refresh token. CSRF-exempt (authenticated by HttpOnly refresh cookie)."""
    raw_refresh = request.cookies.get(_REFRESH_COOKIE)
    if not raw_refresh:
        raise HTTPException(status_code=401, detail="No refresh token")

    result = validate_and_rotate_refresh_token(db, raw_refresh)
    if not result:
        _clear_auth_cookies(response)
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    new_raw_refresh, user = result
    access_token = create_access_token({"sub": user.id, "email": user.email, "role": user.role.value})
    csrf_token = secrets.token_urlsafe(32)

    _set_auth_cookies(response, access_token, new_raw_refresh, csrf_token)
    log_security_event(db, "TOKEN_REFRESHED", f"Token refreshed for {user.email}", "INFO", user_id=user.id)

    return RefreshResponse(
        expires_in=settings.access_token_expire_minutes * 60,
        user=_user_info(user),
    )


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    raw_refresh = request.cookies.get(_REFRESH_COOKIE)
    if raw_refresh:
        revoke_refresh_token(db, raw_refresh)
    _clear_auth_cookies(response)
    log_security_event(db, "LOGOUT", f"User {current_user.email} logged out", "INFO", user_id=current_user.id)
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserInfo)
def me(current_user: User = Depends(get_current_user)):
    return _user_info(current_user)


@router.post("/change-password")
def change_password(
    body: PasswordChangeRequest,
    response: Response,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Change password. Requires current password. Revokes all refresh tokens (forces re-login)."""
    if not verify_password(body.current_password, current_user.hashed_password):
        log_security_event(
            db, "PASSWORD_CHANGE_FAILED",
            f"Wrong current password for {current_user.email}", "WARNING",
            user_id=current_user.id,
        )
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    current_user.hashed_password = hash_password(body.new_password)
    db.commit()

    revoked = revoke_all_refresh_tokens(db, current_user.id)
    _clear_auth_cookies(response)

    log_security_event(
        db, "PASSWORD_CHANGED",
        f"Password changed for {current_user.email} — {revoked} sessions revoked", "WARNING",
        user_id=current_user.id,
    )
    return {"message": "Password changed. Please log in again."}
