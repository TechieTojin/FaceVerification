"""Organisations, invitations and membership onboarding."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.config import settings
from backend.core import crypto
from backend.core.deps import Actor
from backend.core.errors import ApiError
from backend.models.organization import (
    Invitation, InvitationStatus, Membership, MembershipStatus, Organization, OrgStatus,
)
from backend.models.user import RoleEnum, User
from backend.services import email_service, notification_service
from backend.services.audit_service import audit, log_security_event
from backend.services.auth_service import (
    PasswordPolicyError, create_user, get_user_by_email, normalise_email, validate_password_strength,
)
from backend.services.policy_service import ensure_policy_seeded

INVITATION_TTL = timedelta(days=7)
_SLUG = re.compile(r"^[a-z0-9](?:[a-z0-9-]{1,46}[a-z0-9])$")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:48] or "org"


def create_organization(db: Session, actor: Actor, name: str, slug: Optional[str], ip: str) -> Organization:
    name = name.strip()
    if not name:
        raise ApiError(422, "INVALID_NAME", "Organisation name is required")
    slug = (slug or slugify(name)).strip().lower()
    if not _SLUG.match(slug):
        raise ApiError(422, "INVALID_SLUG", "Slug must be 3-48 characters of lowercase letters, digits and hyphens")
    if db.query(Organization).filter(Organization.slug == slug).first():
        raise ApiError(409, "SLUG_TAKEN", "An organisation with this slug already exists")
    org = Organization(name=name[:200], slug=slug, status=OrgStatus.ACTIVE, created_by_user_id=actor.id)
    db.add(org)
    db.flush()
    db.add(Membership(organization_id=org.id, user_id=actor.id, role=RoleEnum.ADMIN, status=MembershipStatus.ACTIVE))
    db.commit()
    ensure_policy_seeded(db, org.id)
    audit(db, actor.id, "ORGANIZATION_CREATED", "organization", org.id, {"name": org.name, "slug": slug}, ip,
          org_id=org.id)
    log_security_event(db, "ORGANIZATION_CREATED", f"{actor.email} created organisation {org.name}", "WARNING",
                       user_id=actor.id, ip_address=ip, org_id=org.id)
    return org


def org_stats(db: Session, org_id: str) -> dict:
    from backend.models.devices import Device
    from backend.models.verification import VerificationSession
    members = db.query(func.count(Membership.id)).filter(Membership.organization_id == org_id).scalar() or 0
    active = (db.query(func.count(Membership.id))
              .filter(Membership.organization_id == org_id, Membership.status == MembershipStatus.ACTIVE).scalar() or 0)
    sessions = (db.query(func.count(VerificationSession.id))
                .filter(VerificationSession.organization_id == org_id).scalar() or 0)
    devices = db.query(func.count(Device.id)).filter(Device.organization_id == org_id).scalar() or 0
    return {"members": members, "active_members": active, "sessions": sessions, "devices": devices}


def org_view(db: Session, org: Organization, role: Optional[RoleEnum] = None) -> dict:
    return {"id": org.id, "name": org.name, "slug": org.slug, "status": org.status, "created_at": org.created_at,
            "role": role.value if role else None, **org_stats(db, org.id)}


def set_status(db: Session, actor: Actor, org_id: str, status: str, ip: str) -> Organization:
    org = db.query(Organization).filter(Organization.id == org_id).first()
    if org is None:
        raise ApiError(404, "NOT_FOUND", "Organisation not found")
    if status == OrgStatus.SUSPENDED and org.id == actor.org_id:
        raise ApiError(409, "ACTIVE_ORG", "Switch to another organisation before suspending this one")
    org.status = status
    db.commit()
    audit(db, actor.id, f"ORGANIZATION_{status}", "organization", org.id, None, ip, org_id=org.id)
    log_security_event(db, f"ORGANIZATION_{status}", f"{actor.email} set {org.name} to {status}", "HIGH",
                       user_id=actor.id, ip_address=ip, org_id=org.id)
    return org


# ── invitations ─────────────────────────────────────────────────────────────

def _invite_link(raw: str) -> str:
    return f"{settings.public_base_url.rstrip('/')}/invite/{raw}"


def create_invitation(db: Session, actor: Actor, email: str, role: RoleEnum, ip: str) -> tuple[Invitation, str, str]:
    """Returns (invitation, link, email_status). The link is shown once to the inviting admin."""
    email = normalise_email(email)
    existing = get_user_by_email(db, email)
    if existing and db.query(Membership).filter(Membership.organization_id == actor.org_id,
                                                Membership.user_id == existing.id).first():
        raise ApiError(409, "ALREADY_MEMBER", "This account is already a member of the organisation")
    if role == RoleEnum.ADMIN and actor.role != RoleEnum.ADMIN:
        raise ApiError(403, "ROLE_NOT_PERMITTED", "Only administrators may invite administrators")
    for old in (db.query(Invitation).filter(Invitation.organization_id == actor.org_id, Invitation.email == email,
                                            Invitation.status == InvitationStatus.PENDING)):
        old.status = InvitationStatus.REVOKED
    raw = crypto.token(32)
    inv = Invitation(organization_id=actor.org_id, email=email, role=role,
                     token_hash=crypto.keyed_hash(raw, "invite"), status=InvitationStatus.PENDING,
                     expires_at=datetime.utcnow() + INVITATION_TTL, created_by_user_id=actor.id)
    db.add(inv)
    db.commit()
    link = _invite_link(raw)
    email_status = "UNAVAILABLE"
    if email_service.available():
        org = db.query(Organization).filter(Organization.id == actor.org_id).first()
        try:
            email_service.send(email, f"Invitation to {org.name} on FaceVerification",
                               f"You have been invited to join {org.name} as {role.value}.\n\n"
                               f"Accept the invitation: {link}\n\nThis link expires in 7 days.")
            email_status = "SENT"
        except email_service.EmailSendError:
            email_status = "FAILED"
    audit(db, actor.id, "INVITATION_CREATED", "invitation", inv.id,
          {"email": email, "role": role.value, "email_status": email_status}, ip, org_id=actor.org_id)
    return inv, link, email_status


def _load_invitation(db: Session, raw: str) -> Invitation:
    inv = db.query(Invitation).filter(Invitation.token_hash == crypto.keyed_hash(raw, "invite")).first()
    if inv is None:
        raise ApiError(404, "INVITATION_NOT_FOUND", "This invitation link is not valid")
    if inv.status == InvitationStatus.PENDING and inv.expires_at < datetime.utcnow():
        inv.status = InvitationStatus.EXPIRED
        db.commit()
    if inv.status != InvitationStatus.PENDING:
        raise ApiError(410, f"INVITATION_{inv.status}", f"This invitation is {inv.status.lower()}")
    return inv


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return (local[:2] + "•" * max(1, len(local) - 2)) + "@" + domain


def invitation_info(db: Session, raw: str) -> dict:
    inv = _load_invitation(db, raw)
    org = db.query(Organization).filter(Organization.id == inv.organization_id).first()
    return {"organization": org.name, "role": inv.role.value, "email": mask_email(inv.email),
            "account_exists": get_user_by_email(db, inv.email) is not None, "expires_at": inv.expires_at}


def accept_invitation(db: Session, raw: str, signed_in: Optional[User], full_name: Optional[str],
                      password: Optional[str], ip: str) -> tuple[User, str]:
    inv = _load_invitation(db, raw)
    user = get_user_by_email(db, inv.email)
    if user is not None:
        if signed_in is None or signed_in.id != user.id:
            raise ApiError(409, "SIGN_IN_REQUIRED", "Sign in as the invited account to accept this invitation")
        m = (db.query(Membership)
             .filter(Membership.organization_id == inv.organization_id, Membership.user_id == user.id).first())
        if m is None:
            db.add(Membership(organization_id=inv.organization_id, user_id=user.id, role=inv.role,
                              status=MembershipStatus.ACTIVE, invited_by_user_id=inv.created_by_user_id))
    else:
        if signed_in is not None:
            raise ApiError(409, "WRONG_ACCOUNT", "This invitation is for a different email address")
        if not full_name or not full_name.strip() or not password:
            raise ApiError(422, "DETAILS_REQUIRED", "Enter your name and choose a password")
        try:
            validate_password_strength(password)
        except PasswordPolicyError as exc:
            raise ApiError(422, "WEAK_PASSWORD", str(exc)) from exc
        user = create_user(db, inv.email, full_name, password, inv.role, commit=False, org_id=inv.organization_id,
                           invited_by=inv.created_by_user_id)
    inv.status = InvitationStatus.ACCEPTED
    inv.accepted_at = datetime.utcnow()
    inv.accepted_user_id = user.id
    db.commit()
    audit(db, user.id, "INVITATION_ACCEPTED", "invitation", inv.id, {"role": inv.role.value}, ip,
          org_id=inv.organization_id)
    if inv.created_by_user_id:
        notification_service.notify(db, inv.organization_id, inv.created_by_user_id, "ACCOUNT_SECURITY",
                                    "Invitation accepted", f"{user.email} joined the organisation as {inv.role.value}.",
                                    link="/organization")
        db.commit()
    return user, inv.organization_id


def revoke_invitation(db: Session, actor: Actor, invitation_id: str, ip: str) -> Invitation:
    inv = (db.query(Invitation)
           .filter(Invitation.id == invitation_id, Invitation.organization_id == actor.org_id).first())
    if inv is None:
        raise ApiError(404, "NOT_FOUND", "Invitation not found")
    if inv.status != InvitationStatus.PENDING:
        raise ApiError(409, "NOT_PENDING", f"Invitation is already {inv.status.lower()}")
    inv.status = InvitationStatus.REVOKED
    db.commit()
    audit(db, actor.id, "INVITATION_REVOKED", "invitation", inv.id, {"email": inv.email}, ip, org_id=actor.org_id)
    return inv


def invitation_view(inv: Invitation) -> dict:
    status = inv.status
    if status == InvitationStatus.PENDING and inv.expires_at < datetime.utcnow():
        status = InvitationStatus.EXPIRED
    return {"id": inv.id, "email": inv.email, "role": inv.role.value, "status": status, "created_at": inv.created_at,
            "expires_at": inv.expires_at, "accepted_at": inv.accepted_at}
