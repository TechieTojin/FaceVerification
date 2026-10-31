"""Organisations: listing, creation (platform admins), profile, invitations."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.core.deps import (
    Actor, client_ip, get_actor, oauth2_scheme, require, require_platform_admin, require_step_up,
)
from backend.core.errors import ApiError
from backend.core.permissions import Permission
from backend.database import get_db
from backend.models.organization import Invitation, Membership, Organization, OrgStatus
from backend.models.user import RoleEnum
from backend.routers.users import check_email
from backend.services import org_service
from backend.services.audit_service import audit

router = APIRouter(prefix="/api/v1/organizations", tags=["organizations"])
public = APIRouter(prefix="/api/v1/invitations", tags=["organizations"])


class OrgCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    slug: Optional[str] = Field(None, max_length=48)


class OrgUpdate(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)


class InviteBody(BaseModel):
    email: str
    role: RoleEnum = RoleEnum.OPERATOR

    _email = field_validator("email")(check_email)


class AcceptBody(BaseModel):
    full_name: Optional[str] = Field(None, max_length=255)
    password: Optional[str] = Field(None, max_length=256)


@router.get("")
def list_orgs(db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    """Platform administrators see every organisation; everyone else sees their own memberships."""
    mine = {m.organization_id: m.role for m in db.query(Membership).filter(Membership.user_id == actor.id)}
    if actor.is_platform_admin:
        rows = db.query(Organization).order_by(Organization.name).all()
    else:
        rows = (db.query(Organization).filter(Organization.id.in_(list(mine)), Organization.status == OrgStatus.ACTIVE)
                .order_by(Organization.name).all())
    return {"organizations": [org_service.org_view(db, o, mine.get(o.id)) for o in rows],
            "active_organization_id": actor.org_id, "can_create": actor.is_platform_admin}


@router.post("", status_code=201)
def create_org(body: OrgCreate, request: Request, db: Session = Depends(get_db),
               actor: Actor = Depends(require_platform_admin), _: Actor = Depends(require_step_up(high_risk=True))):
    org = org_service.create_organization(db, actor, body.name, body.slug, client_ip(request))
    return org_service.org_view(db, org, RoleEnum.ADMIN)


@router.get("/current")
def current(db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.ORG_READ))):
    org = db.query(Organization).filter(Organization.id == actor.org_id).first()
    return org_service.org_view(db, org, actor.role)


@router.patch("/current")
def update_current(body: OrgUpdate, request: Request, db: Session = Depends(get_db),
                   actor: Actor = Depends(require(Permission.ORG_MANAGE))):
    org = db.query(Organization).filter(Organization.id == actor.org_id).first()
    old = org.name
    org.name = body.name.strip()
    db.commit()
    audit(db, actor.id, "ORGANIZATION_UPDATED", "organization", org.id, {"name": [old, org.name]},
          client_ip(request), org_id=org.id)
    return org_service.org_view(db, org, actor.role)


@router.post("/{org_id}/suspend")
def suspend(org_id: str, request: Request, db: Session = Depends(get_db),
            actor: Actor = Depends(require_platform_admin), _: Actor = Depends(require_step_up(high_risk=True))):
    return org_service.org_view(db, org_service.set_status(db, actor, org_id, OrgStatus.SUSPENDED, client_ip(request)))


@router.post("/{org_id}/activate")
def activate(org_id: str, request: Request, db: Session = Depends(get_db),
             actor: Actor = Depends(require_platform_admin)):
    return org_service.org_view(db, org_service.set_status(db, actor, org_id, OrgStatus.ACTIVE, client_ip(request)))


@router.get("/current/invitations")
def list_invitations(db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.ORG_MANAGE))):
    rows = (db.query(Invitation).filter(Invitation.organization_id == actor.org_id)
            .order_by(Invitation.created_at.desc()).limit(200).all())
    return {"invitations": [org_service.invitation_view(i) for i in rows]}


@router.post("/current/invitations", status_code=201)
def invite(body: InviteBody, request: Request, db: Session = Depends(get_db),
           actor: Actor = Depends(require(Permission.ORG_MANAGE))):
    inv, link, email_status = org_service.create_invitation(db, actor, body.email, body.role, client_ip(request))
    return {**org_service.invitation_view(inv), "link": link, "email_status": email_status}


@router.post("/current/invitations/{invitation_id}/revoke")
def revoke(invitation_id: str, request: Request, db: Session = Depends(get_db),
           actor: Actor = Depends(require(Permission.ORG_MANAGE))):
    return org_service.invitation_view(org_service.revoke_invitation(db, actor, invitation_id, client_ip(request)))


def _optional_user(request: Request, db: Session, bearer: Optional[str]):
    from backend.core.deps import _authenticate
    try:
        user, _ = _authenticate(request, db, bearer)
        return user
    except Exception:  # noqa: BLE001 — not signed in
        return None


@public.get("/{token}")
def invitation_info(token: str, db: Session = Depends(get_db)):
    if len(token) > 128:
        raise ApiError(404, "INVITATION_NOT_FOUND", "This invitation link is not valid")
    return org_service.invitation_info(db, token)


@public.post("/{token}/accept")
def accept(token: str, body: AcceptBody, request: Request, response: Response, db: Session = Depends(get_db),
           bearer: Optional[str] = Depends(oauth2_scheme)):
    if len(token) > 128:
        raise ApiError(404, "INVITATION_NOT_FOUND", "This invitation link is not valid")
    signed_in = _optional_user(request, db, bearer)
    user, org_id = org_service.accept_invitation(db, token, signed_in, body.full_name, body.password,
                                                 client_ip(request))
    if signed_in is None:
        from backend.routers.auth import issue_session
        return {"accepted": True, "organization_id": org_id,
                "session": issue_session(db, request, response, user, ["pwd"], org_id)}
    return {"accepted": True, "organization_id": org_id}
