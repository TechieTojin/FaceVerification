"""Authentication: login, refresh rotation, logout, profile, password, device sessions.

Browser clients use HttpOnly cookies (fv_access, fv_refresh) plus a
double-submit CSRF cookie (fv_csrf). API clients may use the bearer token
returned by /login. State-changing cookie requests require X-CSRF-Token.
"""
from __future__ import annotations

import secrets
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.config import settings
from backend.core.deps import ACCESS_COOKIE, Actor, active_membership, client_ip, get_actor, get_current_user
from backend.core.permissions import permissions_for
from backend.database import get_db
from backend.models.auth import RefreshToken
from backend.models.user import User
from backend.services import alert_service
from backend.services.audit_service import audit, log_security_event
from backend.services.auth_service import (
    PasswordPolicyError, active_orgs, authenticate_user, get_user_by_email, check_rate_limit, create_refresh_token,
    current_family, hash_password, pick_org, record_login_attempt, revoke_all_refresh_tokens, revoke_family,
    revoke_refresh_token, session_token, set_family_org, validate_and_rotate_refresh_token,
    validate_password_strength, verify_password,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

_REFRESH_COOKIE = settings.refresh_token_cookie_name
_CSRF_COOKIE = settings.csrf_cookie_name


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=1, max_length=256)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=256)
    new_password: str = Field(..., min_length=1, max_length=256)


class SwitchOrgRequest(BaseModel):
    organization_id: str = Field(..., min_length=1, max_length=36)


def user_info(db: Session, user: User, org_id: Optional[str], amr: Optional[list[str]] = None) -> dict:
    orgs = active_orgs(db, user.id)
    current = next(((o, m) for o, m in orgs if o.id == org_id), None)
    role = current[1].role if current else None
    return {
        "id": user.id, "email": user.email, "full_name": user.full_name,
        "role": role.value if role else None,
        "is_platform_admin": bool(user.is_platform_admin),
        "is_active": user.is_active, "created_at": user.created_at, "last_login_at": user.last_login_at,
        "password_changed_at": user.password_changed_at,
        "permissions": sorted(p.value for p in permissions_for(role)) if role else [],
        "organization": ({"id": current[0].id, "name": current[0].name, "slug": current[0].slug}
                         if current else None),
        "organizations": [{"id": o.id, "name": o.name, "slug": o.slug, "role": m.role.value} for o, m in orgs],
        "amr": amr or [],
    }


def _set_auth_cookies(response: Response, access: str, refresh: str, csrf: str) -> None:
    common = {"secure": settings.cookie_secure, "samesite": settings.cookie_samesite}
    response.set_cookie(ACCESS_COOKIE, access, httponly=True, max_age=settings.access_token_expire_minutes * 60,
                        path="/", **common)
    response.set_cookie(_REFRESH_COOKIE, refresh, httponly=True, max_age=settings.refresh_token_expire_days * 86400,
                        path="/api/v1/auth", **common)
    # The CSRF cookie lives as long as the refresh token so cookie-authenticated
    # requests made right after an access-token refresh still carry it.
    response.set_cookie(_CSRF_COOKIE, csrf, httponly=False, max_age=settings.refresh_token_expire_days * 86400,
                        path="/", **common)


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(ACCESS_COOKIE, path="/")
    response.delete_cookie(_REFRESH_COOKIE, path="/api/v1/auth")
    response.delete_cookie(_CSRF_COOKIE, path="/")


def issue_session(db: Session, request: Request, response: Response, user: User, amr: list[str],
                  org_id: Optional[str] = None) -> dict:
    """Issue access + refresh cookies for a fully authenticated sign-in. Every sign-in path ends here."""
    org = pick_org(db, user, org_id)
    if org is None:
        raise HTTPException(status_code=403, detail="Your account is not a member of any active organisation")
    now = int(time.time())
    access = session_token(user, org, amr, now)
    refresh_raw = create_refresh_token(db, user.id, request.headers.get("user-agent", ""), client_ip(request),
                                       org_id=org, amr=amr)
    _set_auth_cookies(response, access, refresh_raw, secrets.token_urlsafe(32))
    user.last_login_at = datetime.utcnow()
    user.last_org_id = org
    db.commit()
    log_security_event(db, "LOGIN_SUCCESS", f"{user.email} signed in ({'+'.join(amr)})", "INFO", user_id=user.id,
                       ip_address=client_ip(request), org_id=org)
    return {"access_token": access, "token_type": "bearer",
            "expires_in": settings.access_token_expire_minutes * 60, "user": user_info(db, user, org, amr)}


@router.post("/login")
def login(request: Request, body: LoginRequest, response: Response, db: Session = Depends(get_db)):
    ip = client_ip(request)
    allowed, reason, retry_after = check_rate_limit(db, body.email, ip)
    if not allowed:
        if reason == "account_locked":
            locked = get_user_by_email(db, body.email)
            alert_service.raise_for_user(db, locked.id if locked else None, "ACCOUNT_LOCKOUT", "MEDIUM",
                                         f"Login locked after repeated failures (from {ip})", window_minutes=30)
        raise HTTPException(status_code=429, headers={"Retry-After": str(retry_after)},
                            detail=(f"Account temporarily locked. Try again in {retry_after}s."
                                    if reason == "account_locked" else "Too many login attempts. Try again later."))
    user = authenticate_user(db, body.email, body.password)
    record_login_attempt(db, body.email, ip, user is not None)
    if user is None:
        log_security_event(db, "LOGIN_FAILED", f"Failed login for {body.email.lower()[:255]}", "WARNING", ip_address=ip)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    from backend.services import mfa_service
    step = mfa_service.after_password(db, user, ip)
    if not step.done:
        log_security_event(db, "LOGIN_PASSWORD_OK", f"{user.email}: password accepted, second factor required",
                           "INFO", user_id=user.id, ip_address=ip)
        return {"status": step.kind, "pending_token": step.pending_token, "methods": step.methods,
                "enrol_required": step.enrol_required}
    return {"status": "ok", **issue_session(db, request, response, user, ["pwd"])}


@router.post("/refresh")
def refresh(request: Request, response: Response, db: Session = Depends(get_db)):
    raw = request.cookies.get(_REFRESH_COOKIE)
    if not raw:
        raise HTTPException(status_code=401, detail="No refresh token")
    ip = client_ip(request)
    result = validate_and_rotate_refresh_token(db, raw, request.headers.get("user-agent", ""), ip)
    if result.reuse_detected:
        log_security_event(db, "REFRESH_TOKEN_REUSE", "Rotated refresh token presented again; all sessions revoked",
                           "HIGH", user_id=result.reuse_user_id, ip_address=ip)
        alert_service.raise_for_user(db, result.reuse_user_id, "REFRESH_TOKEN_REUSE", "HIGH",
                                     "A previously rotated refresh token was reused; all sessions were revoked")
    if not result.ok:
        _clear_auth_cookies(response)
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    access = session_token(result.user, result.org_id, result.amr, result.auth_time or int(time.time()))
    _set_auth_cookies(response, access, result.raw, secrets.token_urlsafe(32))
    return {"expires_in": settings.access_token_expire_minutes * 60,
            "user": user_info(db, result.user, result.org_id, result.amr)}


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db),
           current_user: User = Depends(get_current_user)):
    raw = request.cookies.get(_REFRESH_COOKIE)
    if raw:
        revoke_refresh_token(db, raw)
    _clear_auth_cookies(response)
    log_security_event(db, "LOGOUT", f"{current_user.email} logged out", "INFO", user_id=current_user.id,
                       ip_address=client_ip(request))
    return {"message": "Logged out"}


@router.get("/me")
def me(db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    return user_info(db, actor.user, actor.org_id, list(actor.amr))


@router.post("/switch-org")
def switch_org(body: SwitchOrgRequest, request: Request, response: Response, db: Session = Depends(get_db),
               actor: Actor = Depends(get_actor)):
    """Move the current sign-in to another organisation the user is an active member of."""
    if active_membership(db, body.organization_id, actor.user.id) is None:
        log_security_event(db, "ORG_SWITCH_DENIED", f"{actor.email} tried to switch to an organisation they do not belong to",
                           "WARNING", user_id=actor.id, ip_address=client_ip(request), org_id=actor.org_id,
                           metadata={"requested": body.organization_id[:36]})
        raise HTTPException(status_code=404, detail="Organisation not found")
    raw = request.cookies.get(_REFRESH_COOKIE)
    set_family_org(db, raw, body.organization_id)
    actor.user.last_org_id = body.organization_id
    db.commit()
    access = session_token(actor.user, body.organization_id, list(actor.amr), actor.auth_time or int(time.time()))
    common = {"secure": settings.cookie_secure, "samesite": settings.cookie_samesite}
    response.set_cookie(ACCESS_COOKIE, access, httponly=True, max_age=settings.access_token_expire_minutes * 60,
                        path="/", **common)
    audit(db, actor.id, "ORG_SWITCHED", "organization", body.organization_id, None, client_ip(request),
          org_id=body.organization_id)
    return {"access_token": access, "user": user_info(db, actor.user, body.organization_id, list(actor.amr))}


@router.post("/change-password")
def change_password(body: PasswordChangeRequest, request: Request, response: Response,
                    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    ip = client_ip(request)
    if not verify_password(body.current_password, current_user.hashed_password):
        log_security_event(db, "PASSWORD_CHANGE_FAILED", f"Wrong current password for {current_user.email}",
                           "WARNING", user_id=current_user.id, ip_address=ip)
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if body.new_password == body.current_password:
        raise HTTPException(status_code=422, detail="New password must differ from the current password")
    try:
        validate_password_strength(body.new_password)
    except PasswordPolicyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    current_user.hashed_password = hash_password(body.new_password)
    current_user.password_changed_at = datetime.utcnow()
    db.commit()
    revoked = revoke_all_refresh_tokens(db, current_user.id)
    _clear_auth_cookies(response)
    audit(db, current_user.id, "PASSWORD_CHANGED", "user", current_user.id, {"sessions_revoked": revoked}, ip)
    log_security_event(db, "PASSWORD_CHANGED", f"Password changed for {current_user.email}; {revoked} sessions revoked",
                       "WARNING", user_id=current_user.id, ip_address=ip)
    return {"message": "Password changed. Please sign in again.", "sessions_revoked": revoked}


@router.get("/sessions")
def list_sessions(request: Request, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Active sign-in sessions (refresh-token families) for the current user."""
    mine = current_family(db, request.cookies.get(_REFRESH_COOKIE))
    now = datetime.utcnow()
    rows = (db.query(RefreshToken)
            .filter(RefreshToken.user_id == current_user.id, RefreshToken.revoked == False,  # noqa: E712
                    RefreshToken.expires_at > now)
            .order_by(RefreshToken.created_at.desc()).all())
    return {"sessions": [{"family_id": r.family_id, "last_rotated_at": r.created_at, "expires_at": r.expires_at,
                          "user_agent": r.user_agent, "ip_address": r.ip_address, "current": r.family_id == mine}
                         for r in rows]}


@router.post("/sessions/{family_id}/revoke")
def revoke_session(family_id: str, request: Request, db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    n = revoke_family(db, current_user.id, family_id)
    if n == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    audit(db, current_user.id, "SESSION_REVOKED", "user", current_user.id, {"family_id": family_id}, client_ip(request))
    return {"revoked": n}


@router.post("/sessions/revoke-others")
def revoke_other_sessions(request: Request, db: Session = Depends(get_db),
                          current_user: User = Depends(get_current_user)):
    mine: Optional[str] = current_family(db, request.cookies.get(_REFRESH_COOKIE))
    q = db.query(RefreshToken).filter(RefreshToken.user_id == current_user.id, RefreshToken.revoked == False)  # noqa: E712
    if mine:
        q = q.filter(RefreshToken.family_id != mine)
    n = q.update({"revoked": True, "revoked_at": datetime.utcnow()}, synchronize_session=False)
    db.commit()
    audit(db, current_user.id, "OTHER_SESSIONS_REVOKED", "user", current_user.id, {"count": n}, client_ip(request))
    return {"revoked": n}
