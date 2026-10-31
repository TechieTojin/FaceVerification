"""User and identity management inside the active organisation.

Accounts are global, so two kinds of action exist:
  * membership actions (role, enable/disable, remove) affect only the caller's organisation;
  * account actions (rename, password reset, disabling or deleting the account) are allowed only
    when the account is *exclusively* managed by the caller's organisation — it has no membership
    elsewhere and is not a platform administrator. An organisation admin can never lock a user out
    of another organisation.
A user who is not a member of the active organisation is reported as "not found".
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from backend.core.deps import Actor, client_ip, get_actor, require, require_step_up
from backend.core.permissions import Permission
from backend.database import get_db
from backend.models.auth import RefreshToken
from backend.models.biometric import BiometricEnrolment, ConsentRecord, EnrolmentRequest, EnrolmentStatus
from backend.models.organization import Membership, MembershipStatus
from backend.models.settings import AuditLog
from backend.models.user import RoleEnum, User
from backend.models.verification import VerificationSession
from backend.services.audit_service import audit, log_security_event
from backend.services.auth_service import (
    PasswordPolicyError, create_user, get_user_by_email, hash_password, normalise_email, revoke_all_refresh_tokens,
    validate_password_strength,
)
from backend.services.enrolment_service import enrolment_status_view

router = APIRouter(prefix="/api/v1/users", tags=["users"])

_EMAIL_HINT = "Enter a valid email address"


def check_email(v: str) -> str:
    v = normalise_email(v)
    local, _, domain = v.partition("@")
    if not local or "." not in domain or " " in v or len(v) > 255:
        raise ValueError(_EMAIL_HINT)
    return v


class UserCreate(BaseModel):
    email: str
    full_name: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=256)
    role: RoleEnum = RoleEnum.OPERATOR

    _email = field_validator("email")(check_email)


class UserUpdate(BaseModel):
    full_name: Optional[str] = Field(None, min_length=1, max_length=255)
    role: Optional[RoleEnum] = None


class PasswordReset(BaseModel):
    new_password: str = Field(..., min_length=1, max_length=256)


def _enrolment(db: Session, org_id: str, user_id: str) -> Optional[BiometricEnrolment]:
    return (db.query(BiometricEnrolment)
            .filter(BiometricEnrolment.organization_id == org_id, BiometricEnrolment.user_id == user_id).first())


def membership_of(db: Session, org_id: str, user_id: str) -> Optional[Membership]:
    return db.query(Membership).filter(Membership.organization_id == org_id, Membership.user_id == user_id).first()


def is_exclusive(db: Session, org_id: str, u: User) -> bool:
    """True when the account belongs only to `org_id` (so the org may manage the account itself)."""
    if u.is_platform_admin:
        return False
    other = (db.query(Membership)
             .filter(Membership.user_id == u.id, Membership.organization_id != org_id).count())
    return other == 0


def user_view(db: Session, u: User, m: Membership, full: bool = True) -> dict:
    enr = _enrolment(db, m.organization_id, u.id)
    st = enrolment_status_view(enr)
    active = bool(u.is_active and m.status == MembershipStatus.ACTIVE)
    base = {"id": u.id, "email": u.email, "full_name": u.full_name, "role": m.role.value, "is_active": active,
            "enrolment_status": st["status"], "enrolment_usable": st["usable"],
            "reenrolment_requested": st["reenrolment_requested"]}
    if full:
        base.update({"created_at": u.created_at, "updated_at": u.updated_at, "last_login_at": u.last_login_at,
                     "disabled_at": u.disabled_at, "consent_given": st["consent_given"],
                     "membership_status": m.status, "account_active": u.is_active,
                     "member_since": m.created_at, "managed_here": is_exclusive(db, m.organization_id, u)})
    return base


def _member_query(db: Session, org_id: str):
    return (db.query(User, Membership).join(Membership, Membership.user_id == User.id)
            .filter(Membership.organization_id == org_id))


def _load_member(db: Session, actor: Actor, user_id: str) -> tuple[User, Membership]:
    row = _member_query(db, actor.org_id).filter(User.id == user_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    return row


def _active_admins(db: Session, org_id: str) -> int:
    return (_member_query(db, org_id)
            .filter(Membership.role == RoleEnum.ADMIN, Membership.status == MembershipStatus.ACTIVE,
                    User.is_active == True).count())  # noqa: E712


def _require_exclusive(db: Session, actor: Actor, u: User, action: str) -> None:
    if not is_exclusive(db, actor.org_id, u):
        raise HTTPException(status_code=409, detail=f"This account also belongs to another organisation, so it "
                                                    f"cannot be {action} here. Manage the membership instead.")


@router.get("/directory")
def directory(search: Optional[str] = None, enrolled_only: bool = False, limit: int = Query(50, ge=1, le=200),
              db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.USERS_DIRECTORY))):
    """Minimal list of active members for choosing a verification/enrolment subject."""
    q = _member_query(db, actor.org_id).filter(User.is_active == True,  # noqa: E712
                                               Membership.status == MembershipStatus.ACTIVE)
    if search:
        term = f"%{search.strip()[:100].replace('%', '').replace('_', '')}%"
        q = q.filter(or_(User.email.ilike(term), User.full_name.ilike(term)))
    if enrolled_only:
        q = q.join(BiometricEnrolment, (BiometricEnrolment.user_id == User.id)
                   & (BiometricEnrolment.organization_id == actor.org_id)).filter(
            BiometricEnrolment.status == EnrolmentStatus.ACTIVE, BiometricEnrolment.consent_given == True)  # noqa: E712
    rows = q.order_by(User.full_name).limit(limit).all()
    return {"users": [user_view(db, u, m, full=False) for u, m in rows]}


@router.get("")
def list_users(search: Optional[str] = None, role: Optional[RoleEnum] = None,
               status: Optional[Literal["active", "disabled"]] = None,
               enrolment: Optional[Literal["ACTIVE", "NOT_ENROLLED", "REVOKED", "WITHDRAWN", "REENROLMENT_REQUIRED"]] = None,
               sort: Literal["name", "email", "created", "last_login"] = "name",
               page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
               db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.USERS_READ))):
    q = _member_query(db, actor.org_id)
    if search:
        term = f"%{search.strip()[:100].replace('%', '').replace('_', '')}%"
        q = q.filter(or_(User.email.ilike(term), User.full_name.ilike(term)))
    if role:
        q = q.filter(Membership.role == role)
    if status == "active":
        q = q.filter(User.is_active == True, Membership.status == MembershipStatus.ACTIVE)  # noqa: E712
    elif status == "disabled":
        q = q.filter(or_(User.is_active == False, Membership.status != MembershipStatus.ACTIVE))  # noqa: E712
    if enrolment:
        join_on = (BiometricEnrolment.user_id == User.id) & (BiometricEnrolment.organization_id == actor.org_id)
        if enrolment == "NOT_ENROLLED":
            q = q.outerjoin(BiometricEnrolment, join_on).filter(
                or_(BiometricEnrolment.id.is_(None), BiometricEnrolment.status == "NOT_ENROLLED"))
        else:
            q = q.join(BiometricEnrolment, join_on).filter(BiometricEnrolment.status == enrolment)
    order = {"name": User.full_name, "email": User.email, "created": User.created_at.desc(),
             "last_login": User.last_login_at.desc()}[sort]
    total = q.count()
    rows = q.order_by(order).offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size, "users": [user_view(db, u, m) for u, m in rows]}


@router.post("", status_code=201)
def create(body: UserCreate, request: Request, db: Session = Depends(get_db),
           actor: Actor = Depends(require(Permission.USERS_MANAGE))):
    if get_user_by_email(db, body.email):
        raise HTTPException(status_code=409, detail="An account with this email already exists. "
                                                    "Invite it to this organisation instead.")
    try:
        validate_password_strength(body.password)
    except PasswordPolicyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    u = create_user(db, body.email, body.full_name, body.password, body.role, org_id=actor.org_id,
                    invited_by=actor.id)
    audit(db, actor.id, "USER_CREATED", "user", u.id, {"email": u.email, "role": body.role.value}, client_ip(request),
          org_id=actor.org_id)
    return user_view(db, u, membership_of(db, actor.org_id, u.id))


@router.get("/{user_id}")
def get_user(user_id: str, db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    if actor.id != user_id and not actor.can(Permission.USERS_READ):
        raise HTTPException(status_code=403, detail="You do not have permission to view this user")
    u, m = _load_member(db, actor, user_id)
    view = user_view(db, u, m)
    view["enrolment"] = enrolment_status_view(_enrolment(db, actor.org_id, u.id))
    view["verification_count"] = db.query(VerificationSession).filter(
        VerificationSession.organization_id == actor.org_id,
        VerificationSession.target_user_id == u.id, VerificationSession.purpose == "VERIFY").count()
    view["pending_requests"] = db.query(EnrolmentRequest).filter(
        EnrolmentRequest.organization_id == actor.org_id,
        EnrolmentRequest.user_id == u.id, EnrolmentRequest.status == "PENDING").count()
    return view


@router.patch("/{user_id}")
def update(user_id: str, body: UserUpdate, request: Request, db: Session = Depends(get_db),
           actor: Actor = Depends(require(Permission.USERS_MANAGE))):
    u, m = _load_member(db, actor, user_id)
    changes: dict = {}
    if body.full_name is not None and body.full_name.strip() != u.full_name:
        _require_exclusive(db, actor, u, "renamed")
        changes["full_name"] = [u.full_name, body.full_name.strip()]
        u.full_name = body.full_name.strip()
    if body.role is not None and body.role != m.role:
        if u.id == actor.id:
            raise HTTPException(status_code=409, detail="You cannot change your own role")
        if m.role == RoleEnum.ADMIN and m.status == MembershipStatus.ACTIVE and _active_admins(db, actor.org_id) <= 1:
            raise HTTPException(status_code=409, detail="Cannot demote the last active administrator")
        changes["role"] = [m.role.value, body.role.value]
        m.role = body.role
    if not changes:
        return user_view(db, u, m)
    db.commit()
    ip = client_ip(request)
    audit(db, actor.id, "USER_UPDATED", "user", u.id, changes, ip, org_id=actor.org_id)
    if "role" in changes:
        log_security_event(db, "ROLE_CHANGED", f"{u.email}: {changes['role'][0]} → {changes['role'][1]}", "WARNING",
                           user_id=actor.id, ip_address=ip, metadata={"target_user_id": u.id}, org_id=actor.org_id)
        from backend.services import notification_service
        notification_service.notify(db, actor.org_id, u.id, "ACCOUNT_SECURITY", "Your role changed",
                                    f"Your role is now {body.role.value}.", link="/profile", severity="WARNING")
        db.commit()
    return user_view(db, u, m)


def _set_active(db: Session, actor: Actor, user_id: str, active: bool, ip: str) -> dict:
    u, m = _load_member(db, actor, user_id)
    if not active and u.id == actor.id:
        raise HTTPException(status_code=409, detail="You cannot disable your own account")
    if (not active and m.role == RoleEnum.ADMIN and m.status == MembershipStatus.ACTIVE and u.is_active
            and _active_admins(db, actor.org_id) <= 1):
        raise HTTPException(status_code=409, detail="Cannot disable the last active administrator")
    exclusive = is_exclusive(db, actor.org_id, u)
    target_status = MembershipStatus.ACTIVE if active else MembershipStatus.DISABLED
    if m.status == target_status and (not exclusive or u.is_active == active):
        return user_view(db, u, m)
    m.status = target_status
    if exclusive:
        u.is_active = active
        u.disabled_at = None if active else datetime.utcnow()
    db.commit()
    if not active and exclusive:
        revoke_all_refresh_tokens(db, u.id)
    action = "USER_ENABLED" if active else "USER_DISABLED"
    audit(db, actor.id, action, "user", u.id, {"scope": "account" if exclusive else "membership"}, ip,
          org_id=actor.org_id)
    log_security_event(db, action, u.email, "WARNING", user_id=actor.id, ip_address=ip,
                       metadata={"target_user_id": u.id}, org_id=actor.org_id)
    return user_view(db, u, m)


@router.post("/{user_id}/disable")
def disable(user_id: str, request: Request, db: Session = Depends(get_db),
            actor: Actor = Depends(require(Permission.USERS_MANAGE))):
    return _set_active(db, actor, user_id, False, client_ip(request))


@router.post("/{user_id}/enable")
def enable(user_id: str, request: Request, db: Session = Depends(get_db),
           actor: Actor = Depends(require(Permission.USERS_MANAGE))):
    return _set_active(db, actor, user_id, True, client_ip(request))


@router.post("/{user_id}/reset-password")
def reset_password(user_id: str, body: PasswordReset, request: Request, db: Session = Depends(get_db),
                   actor: Actor = Depends(require(Permission.USERS_MANAGE))):
    u, _ = _load_member(db, actor, user_id)
    _require_exclusive(db, actor, u, "reset")
    try:
        validate_password_strength(body.new_password)
    except PasswordPolicyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    u.hashed_password = hash_password(body.new_password)
    u.password_changed_at = datetime.utcnow()
    db.commit()
    revoked = revoke_all_refresh_tokens(db, u.id)
    ip = client_ip(request)
    audit(db, actor.id, "PASSWORD_RESET", "user", u.id, {"sessions_revoked": revoked}, ip, org_id=actor.org_id)
    log_security_event(db, "PASSWORD_RESET", f"Password reset for {u.email} by administrator", "WARNING",
                       user_id=actor.id, ip_address=ip, metadata={"target_user_id": u.id}, org_id=actor.org_id)
    return {"message": "Password reset", "sessions_revoked": revoked}


@router.post("/{user_id}/reset-mfa")
def reset_mfa(user_id: str, request: Request, db: Session = Depends(get_db),
              actor: Actor = Depends(require(Permission.USERS_MANAGE)), _: Actor = Depends(require_step_up())):
    """Account recovery: remove passkeys, authenticator apps and recovery codes of a member who lost them.

    Only for accounts managed exclusively by this organisation; never for your own account.
    """
    u, _m = _load_member(db, actor, user_id)
    if u.id == actor.id:
        raise HTTPException(status_code=409, detail="Use Account security to manage your own sign-in factors")
    _require_exclusive(db, actor, u, "reset")
    from backend.services import mfa_service
    removed = mfa_service.admin_reset_factors(db, actor.id, u, client_ip(request), actor.org_id)
    revoke_all_refresh_tokens(db, u.id)
    return {"removed": removed}


@router.delete("/{user_id}", status_code=204)
def delete(user_id: str, request: Request, db: Session = Depends(get_db),
           actor: Actor = Depends(require(Permission.USERS_MANAGE))):
    u, m = _load_member(db, actor, user_id)
    if u.id == actor.id:
        raise HTTPException(status_code=409, detail="You cannot delete your own account")
    if m.role == RoleEnum.ADMIN and _active_admins(db, actor.org_id) <= 1:
        raise HTTPException(status_code=409, detail="Cannot delete the last administrator")
    org = actor.org_id
    has_history = (db.query(VerificationSession)
                   .filter(VerificationSession.organization_id == org,
                           or_(VerificationSession.target_user_id == u.id,
                               VerificationSession.created_by_user_id == u.id)).first()
                   or db.query(AuditLog).filter(AuditLog.organization_id == org, AuditLog.actor_user_id == u.id).first())
    if has_history:
        raise HTTPException(status_code=409, detail="This user has verification or audit history. Disable the "
                                                    "account instead so the audit trail stays intact.")
    email = u.email
    db.query(BiometricEnrolment).filter(BiometricEnrolment.organization_id == org,
                                        BiometricEnrolment.user_id == u.id).delete(synchronize_session=False)
    db.query(EnrolmentRequest).filter(EnrolmentRequest.organization_id == org,
                                      EnrolmentRequest.user_id == u.id).delete(synchronize_session=False)
    db.query(ConsentRecord).filter(ConsentRecord.organization_id == org,
                                   ConsentRecord.user_id == u.id).delete(synchronize_session=False)
    exclusive = is_exclusive(db, org, u)
    db.delete(m)
    if exclusive:
        db.query(RefreshToken).filter(RefreshToken.user_id == u.id).delete(synchronize_session=False)
        from backend.models.verification import SecurityEvent
        db.query(SecurityEvent).filter(SecurityEvent.user_id == u.id).update({"user_id": None}, synchronize_session=False)
        db.flush()
        db.delete(u)
    db.commit()
    audit(db, actor.id, "USER_DELETED" if exclusive else "MEMBER_REMOVED", "user", user_id, {"email": email},
          client_ip(request), org_id=org)


@router.get("/{user_id}/audit")
def user_audit(user_id: str, limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db),
               actor: Actor = Depends(require(Permission.USERS_READ))):
    _load_member(db, actor, user_id)
    rows = (db.query(AuditLog)
            .filter(AuditLog.organization_id == actor.org_id,
                    or_(AuditLog.target_id == user_id, AuditLog.actor_user_id == user_id))
            .order_by(AuditLog.timestamp.desc()).limit(limit).all())
    ids = {r.actor_user_id for r in rows if r.actor_user_id}
    names = dict(db.query(User.id, User.email).filter(User.id.in_(ids)).all()) if ids else {}
    return {"entries": [{"id": r.id, "timestamp": r.timestamp, "action": r.action, "actor": names.get(r.actor_user_id),
                         "actor_user_id": r.actor_user_id, "target_type": r.target_type, "target_id": r.target_id,
                         "details": json.loads(r.details_json) if r.details_json else None} for r in rows]}


@router.get("/stats/summary")
def user_stats(db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.USERS_READ))):
    by_role = dict(db.query(Membership.role, func.count())
                   .filter(Membership.organization_id == actor.org_id).group_by(Membership.role).all())
    return {"by_role": {k.value: v for k, v in by_role.items()}, "total": sum(by_role.values())}
