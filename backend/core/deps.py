"""Shared FastAPI dependencies: authentication, tenant context and permission checks.

Every authenticated browser/API-token request resolves to an `Actor`:
  * the user account (global),
  * the *active organisation* — taken from the signed `org` claim of the access
    token and re-validated on every request against an ACTIVE membership in an
    ACTIVE organisation (a client cannot pick an organisation it does not belong to),
  * the membership role and its permissions in that organisation,
  * how the session was authenticated (`amr`) and when step-up last happened.
"""
from __future__ import annotations

import calendar
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.core.errors import StepUpRequired
from backend.core.permissions import Permission, permissions_for
from backend.database import get_db
from backend.models.organization import Membership, MembershipStatus, Organization, OrgStatus
from backend.models.user import RoleEnum, User
from backend.services.audit_service import log_security_event
from backend.services.auth_service import decode_token, get_user_by_id

ACCESS_COOKIE = "fv_access"
STRONG_METHODS = frozenset({"webauthn", "totp", "otp", "rc"})

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


@dataclass
class Actor:
    """The authenticated principal of a request, bound to one organisation."""

    kind: str                       # USER / PAIRING / LOGIN / API / DEVICE
    org_id: str
    ref: str                        # owner reference: user id, pairing id, challenge id, client id, device id
    user: Optional[User] = None
    role: Optional[RoleEnum] = None
    permissions: frozenset = field(default_factory=frozenset)
    amr: tuple[str, ...] = ()
    auth_time: Optional[int] = None
    step_up_at: Optional[int] = None
    step_up_amr: tuple[str, ...] = ()
    is_platform_admin: bool = False
    label: str = ""

    @property
    def id(self) -> Optional[str]:
        return self.user.id if self.user is not None else None

    @property
    def email(self) -> str:
        return self.user.email if self.user is not None else (self.label or f"{self.kind.lower()}:{self.ref[:8]}")

    def can(self, perm: Permission) -> bool:
        return perm in self.permissions

    def stepped_up_within(self, seconds: int) -> bool:
        now = int(time.time())
        if self.step_up_at and now - self.step_up_at <= seconds:
            return True
        # A sign-in that itself used a strong factor counts as fresh step-up.
        return bool(self.auth_time and now - self.auth_time <= seconds and STRONG_METHODS & set(self.amr))


def client_ip(request: Request) -> str:
    """Direct TCP peer address. Forwarded headers are not trusted."""
    return request.client.host if request.client else "unknown"


def active_membership(db: Session, org_id: Optional[str], user_id: str) -> Optional[Membership]:
    if not org_id:
        return None
    return (db.query(Membership)
            .join(Organization, Organization.id == Membership.organization_id)
            .filter(Membership.organization_id == org_id, Membership.user_id == user_id,
                    Membership.status == MembershipStatus.ACTIVE, Organization.status == OrgStatus.ACTIVE)
            .first())


def membership_role(db: Session, org_id: str, user_id: str) -> Optional[RoleEnum]:
    m = (db.query(Membership)
         .filter(Membership.organization_id == org_id, Membership.user_id == user_id,
                 Membership.status == MembershipStatus.ACTIVE).first())
    return m.role if m else None


def _token_from(request: Request, bearer: Optional[str]) -> Optional[str]:
    return request.cookies.get(ACCESS_COOKIE) or bearer


def _authenticate(request: Request, db: Session, bearer: Optional[str]) -> tuple[User, dict]:
    exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated",
                        headers={"WWW-Authenticate": "Bearer"})
    token = _token_from(request, bearer)
    if not token:
        raise exc
    payload = decode_token(token)
    if not payload or not payload.get("sub"):
        raise exc
    user = get_user_by_id(db, payload["sub"])
    if not user or not user.is_active:
        raise exc
    # Tokens issued before a password change are invalid.
    iat = payload.get("iat")
    if iat and user.password_changed_at and int(iat) < calendar.timegm(user.password_changed_at.utctimetuple()):
        raise exc
    return user, payload


def get_current_user(request: Request, db: Session = Depends(get_db),
                     bearer: Optional[str] = Depends(oauth2_scheme)) -> User:
    """Authenticated account without organisation context (account-level endpoints only)."""
    user, _ = _authenticate(request, db, bearer)
    request.state.user = user
    return user


def get_actor(request: Request, db: Session = Depends(get_db),
              bearer: Optional[str] = Depends(oauth2_scheme)) -> Actor:
    user, payload = _authenticate(request, db, bearer)
    org_id = payload.get("org")
    m = active_membership(db, org_id, user.id)
    if m is None:
        # Membership removed/disabled or organisation suspended since the token was issued.
        raise HTTPException(status_code=401, detail="Your access to this organisation has ended",
                            headers={"WWW-Authenticate": "Bearer"})
    actor = Actor(kind="USER", org_id=m.organization_id, ref=user.id, user=user, role=m.role,
                  permissions=permissions_for(m.role), amr=tuple(payload.get("amr") or ()),
                  auth_time=payload.get("auth_time"), step_up_at=payload.get("su"),
                  step_up_amr=tuple(payload.get("su_amr") or ()),
                  is_platform_admin=bool(user.is_platform_admin))
    request.state.user = user
    request.state.actor = actor
    return actor


def require(*perms: Permission) -> Callable[..., Actor]:
    """Dependency factory: the caller must hold every listed permission in the active organisation."""

    def _dep(request: Request, actor: Actor = Depends(get_actor), db: Session = Depends(get_db)) -> Actor:
        missing = [p for p in perms if not actor.can(p)]
        if missing:
            log_security_event(db, "PERMISSION_DENIED",
                               f"{actor.role.value if actor.role else actor.kind} denied "
                               f"{', '.join(p.value for p in missing)} on {request.url.path}",
                               "WARNING", user_id=actor.id, ip_address=client_ip(request), org_id=actor.org_id,
                               metadata={"method": request.method, "path": request.url.path})
            raise HTTPException(status_code=403, detail="You do not have permission to perform this action")
        return actor

    return _dep


def require_step_up(high_risk: bool = False) -> Callable[..., Actor]:
    """Sensitive actions need a recent step-up (or a sign-in that used a strong factor).

    The validity window is the organisation's `step_up_max_age_seconds` policy. For high-risk
    actions, an organisation can additionally require that the step-up combined a live face
    verification with an independent factor (`high_risk_requires_face`).
    """

    def _dep(actor: Actor = Depends(get_actor), db: Session = Depends(get_db)) -> Actor:
        from backend.services.policy_service import get_active_policy
        policy = get_active_policy(db, actor.org_id)
        max_age = int(policy.step_up_max_age_seconds)
        if high_risk and policy.high_risk_requires_face:
            fresh = bool(actor.step_up_at and int(time.time()) - actor.step_up_at <= max_age)
            if not (fresh and "face" in actor.step_up_amr and len(set(actor.step_up_amr) - {"face"}) >= 1):
                raise StepUpRequired("This action needs a live face check plus a second factor", level="face_and_factor")
            return actor
        if not actor.stepped_up_within(max_age):
            raise StepUpRequired()
        return actor

    return _dep


def require_platform_admin(actor: Actor = Depends(get_actor)) -> Actor:
    if not actor.is_platform_admin:
        raise HTTPException(status_code=403, detail="Platform administrator access required")
    return actor
