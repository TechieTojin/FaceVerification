"""Passwordless sign-in (passkeys), multi-factor sign-in, factor management and step-up.

Sign-in paths (all end in `auth.issue_session`):
  * password → [second factor if enrolled or required by policy]
  * passkey (WebAuthn, user verification required) → [live face check if organisation policy requires]
  * pending MFA enrolment: password → set up authenticator app → session
A live face check never starts a sign-in; it is only reachable with the token
returned by a successful passkey assertion.
"""
from __future__ import annotations

import time
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.config import settings
from backend.core.deps import ACCESS_COOKIE, Actor, client_ip, get_actor, require_step_up
from backend.core.errors import ApiError
from backend.database import get_db
from backend.models.mfa import ChallengeKind, WebAuthnCredential
from backend.models.user import User
from backend.routers.auth import issue_session, user_info
from backend.services import mfa_service
from backend.services.audit_service import audit
from backend.services.auth_service import session_token

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

Method = Literal["totp", "recovery_code", "email_otp", "webauthn"]


class PendingBody(BaseModel):
    pending_token: str = Field(..., min_length=10, max_length=200)


class MfaVerifyBody(PendingBody):
    method: Method
    code: str = Field("", max_length=32)
    challenge_id: str = Field("", max_length=36)
    credential: Optional[dict[str, Any]] = None


class CredentialBody(BaseModel):
    challenge_id: str = Field(..., max_length=36)
    credential: dict[str, Any]


class RegisterBody(CredentialBody):
    name: str = Field("Passkey", max_length=100)


class RenameBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class TotpConfirmBody(BaseModel):
    factor_id: str = Field(..., max_length=36)
    code: str = Field(..., max_length=12)


class PendingTotpConfirm(PendingBody):
    factor_id: str = Field(..., max_length=36)
    code: str = Field(..., max_length=12)


class StepUpBody(BaseModel):
    method: Literal["password", "totp", "recovery_code", "webauthn"]
    password: str = Field("", max_length=256)
    code: str = Field("", max_length=32)
    challenge_id: str = Field("", max_length=36)
    credential: Optional[dict[str, Any]] = None
    face_token: str = Field("", max_length=200)


def _pending_user(db: Session, row) -> User:
    user = db.query(User).filter(User.id == row.user_id).first()
    if user is None or not user.is_active:
        raise ApiError(401, "INVALID_PENDING_TOKEN", "Sign-in step is not valid. Start again.")
    return user


# ── multi-factor sign-in ────────────────────────────────────────────────────

@router.post("/mfa/webauthn/options")
def mfa_webauthn_options(body: PendingBody, request: Request, db: Session = Depends(get_db)):
    row = mfa_service.load_pending(db, body.pending_token, ChallengeKind.MFA_PENDING)
    user = _pending_user(db, row)
    if not mfa_service.active_credentials(db, user.id):
        raise ApiError(409, "NO_PASSKEY", "No passkey is registered for this account")
    return mfa_service.authentication_options(db, ChallengeKind.WEBAUTHN_STEP_UP, client_ip(request), user=user)


@router.post("/mfa/email-otp")
def mfa_email_otp(body: PendingBody, request: Request, db: Session = Depends(get_db)):
    row = mfa_service.load_pending(db, body.pending_token, ChallengeKind.MFA_PENDING)
    mfa_service.send_email_otp(db, _pending_user(db, row), row, client_ip(request))
    return {"sent": True}


@router.post("/mfa/verify")
def mfa_verify(body: MfaVerifyBody, request: Request, response: Response, db: Session = Depends(get_db)):
    row = mfa_service.load_pending(db, body.pending_token, ChallengeKind.MFA_PENDING)
    user = _pending_user(db, row)
    amr = mfa_service.complete_mfa(db, row, user, body.method, client_ip(request), code=body.code,
                                   challenge_id=body.challenge_id, credential=body.credential)
    return {"status": "ok", **issue_session(db, request, response, user, amr)}


@router.post("/mfa/enrol/totp")
def mfa_enrol_totp(body: PendingBody, db: Session = Depends(get_db)):
    """Policy requires MFA but the account has none: set up an authenticator app during sign-in."""
    row = mfa_service.load_pending(db, body.pending_token, ChallengeKind.MFA_PENDING)
    if not mfa_service._data(row).get("enrol_required"):
        raise ApiError(409, "NOT_REQUIRED", "This account already has a second factor")
    factor, secret, uri = mfa_service.begin_totp(db, _pending_user(db, row))
    return {"factor_id": factor.id, "secret": secret, "otpauth_uri": uri}


@router.post("/mfa/enrol/totp/confirm")
def mfa_enrol_totp_confirm(body: PendingTotpConfirm, request: Request, response: Response,
                           db: Session = Depends(get_db)):
    row = mfa_service.load_pending(db, body.pending_token, ChallengeKind.MFA_PENDING)
    if not mfa_service._data(row).get("enrol_required"):
        raise ApiError(409, "NOT_REQUIRED", "This account already has a second factor")
    user = _pending_user(db, row)
    ip = client_ip(request)
    try:
        _, codes = mfa_service.confirm_totp(db, user, body.factor_id, body.code, ip, None)
    except ApiError:
        mfa_service._failed_attempt(db, row, user, "authenticator setup", ip)
        raise
    mfa_service._consume(db, row)
    return {"status": "ok", "recovery_codes": codes, **issue_session(db, request, response, user, ["pwd", "totp"])}


# ── passkey (passwordless) sign-in ──────────────────────────────────────────

@router.post("/passkey/options")
def passkey_options(request: Request, db: Session = Depends(get_db)):
    """Usernameless: the browser offers the passkeys it holds for this site."""
    return mfa_service.authentication_options(db, ChallengeKind.WEBAUTHN_LOGIN, client_ip(request))


@router.post("/passkey/verify")
def passkey_verify(body: CredentialBody, request: Request, response: Response, db: Session = Depends(get_db)):
    ip = client_ip(request)
    user, cred = mfa_service.verify_assertion(db, body.challenge_id, ChallengeKind.WEBAUTHN_LOGIN, body.credential, ip)
    step = mfa_service.after_passkey(db, user, ip)
    if not step.done:
        return {"status": "face_required", "pending_token": step.pending_token, "user_name": user.full_name}
    return {"status": "ok", **issue_session(db, request, response, user, ["webauthn"], step.org_id)}


@router.post("/login/face/session", status_code=201)
def login_face_session(body: PendingBody, request: Request, db: Session = Depends(get_db)):
    return mfa_service.start_face_session(db, body.pending_token, client_ip(request))


@router.post("/login/face/complete")
def login_face_complete(body: PendingBody, request: Request, response: Response, db: Session = Depends(get_db)):
    user, row, amr = mfa_service.face_result(db, body.pending_token, client_ip(request))
    if mfa_service._data(row).get("purpose") == "step_up":
        raise ApiError(409, "WRONG_FLOW", "This face check belongs to a step-up confirmation")
    return {"status": "ok", **issue_session(db, request, response, user, amr, row.organization_id)}


# ── account security: passkeys, authenticator apps, recovery codes ─────────

@router.get("/security")
def security_overview(db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    return {**mfa_service.security_overview(db, actor.user),
            "webauthn": {"rp_id": settings.webauthn_rp_id, "origins": settings.webauthn_origins},
            "step_up_valid": actor.stepped_up_within(600)}


@router.post("/passkeys/register/options")
def passkey_register_options(request: Request, db: Session = Depends(get_db),
                             actor: Actor = Depends(require_step_up())):
    return mfa_service.registration_options(db, actor.user, client_ip(request))


@router.post("/passkeys/register/verify", status_code=201)
def passkey_register_verify(body: RegisterBody, request: Request, db: Session = Depends(get_db),
                            actor: Actor = Depends(get_actor)):
    cred = mfa_service.register_credential(db, actor.user, body.challenge_id, body.credential, body.name,
                                           client_ip(request), actor.org_id)
    return {"id": cred.id, "name": cred.name, "created_at": cred.created_at}


@router.patch("/passkeys/{credential_id}")
def passkey_rename(credential_id: str, body: RenameBody, db: Session = Depends(get_db),
                   actor: Actor = Depends(get_actor)):
    cred = (db.query(WebAuthnCredential)
            .filter(WebAuthnCredential.id == credential_id, WebAuthnCredential.user_id == actor.id,
                    WebAuthnCredential.revoked_at.is_(None)).first())
    if cred is None:
        raise ApiError(404, "NOT_FOUND", "Passkey not found")
    cred.name = body.name.strip()
    db.commit()
    return {"id": cred.id, "name": cred.name}


@router.delete("/passkeys/{credential_id}")
def passkey_delete(credential_id: str, request: Request, db: Session = Depends(get_db),
                   actor: Actor = Depends(require_step_up())):
    mfa_service.revoke_credential(db, actor.user, credential_id, client_ip(request), actor.org_id)
    return {"revoked": True}


@router.post("/totp/setup")
def totp_setup(db: Session = Depends(get_db), actor: Actor = Depends(require_step_up())):
    factor, secret, uri = mfa_service.begin_totp(db, actor.user)
    return {"factor_id": factor.id, "secret": secret, "otpauth_uri": uri}


@router.post("/totp/confirm")
def totp_confirm(body: TotpConfirmBody, request: Request, db: Session = Depends(get_db),
                 actor: Actor = Depends(get_actor)):
    factor, codes = mfa_service.confirm_totp(db, actor.user, body.factor_id, body.code, client_ip(request),
                                             actor.org_id)
    return {"factor_id": factor.id, "recovery_codes": codes}


@router.delete("/factors/{factor_id}")
def factor_delete(factor_id: str, request: Request, db: Session = Depends(get_db),
                  actor: Actor = Depends(require_step_up())):
    mfa_service.remove_factor(db, actor.user, factor_id, client_ip(request), actor.org_id)
    return {"removed": True}


@router.post("/recovery-codes")
def recovery_codes(request: Request, db: Session = Depends(get_db), actor: Actor = Depends(require_step_up())):
    return {"recovery_codes": mfa_service.regenerate_recovery_codes(db, actor.user, client_ip(request), actor.org_id)}


# ── step-up ─────────────────────────────────────────────────────────────────

@router.post("/step-up/webauthn/options")
def step_up_options(request: Request, db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    if not mfa_service.active_credentials(db, actor.id):
        raise ApiError(409, "NO_PASSKEY", "No passkey is registered for this account")
    return mfa_service.authentication_options(db, ChallengeKind.WEBAUTHN_STEP_UP, client_ip(request), user=actor.user)


@router.post("/step-up/face/session", status_code=201)
def step_up_face_session(request: Request, db: Session = Depends(get_db), actor: Actor = Depends(get_actor)):
    ip = client_ip(request)
    token = mfa_service.begin_face_step_up(db, actor.user, actor.org_id, ip)
    view = mfa_service.start_face_session(db, token, ip)
    return {"face_token": token, "session": view}


@router.post("/step-up")
def step_up(body: StepUpBody, request: Request, response: Response, db: Session = Depends(get_db),
            actor: Actor = Depends(get_actor)):
    ip = client_ip(request)
    methods = mfa_service.step_up(db, actor.user, body.method, ip, password=body.password, code=body.code,
                                  challenge_id=body.challenge_id, credential=body.credential)
    if body.face_token:
        user, row, _ = mfa_service.face_result(db, body.face_token, ip)
        if user.id != actor.id or row.organization_id != actor.org_id or mfa_service._data(row).get("purpose") != "step_up":
            raise ApiError(403, "FACE_CHECK_MISMATCH", "The face check belongs to another sign-in")
        methods = ["face"] + methods
    now = int(time.time())
    access = session_token(actor.user, actor.org_id, list(actor.amr), actor.auth_time or now, step_up_at=now,
                           step_up_amr=methods)
    common = {"secure": settings.cookie_secure, "samesite": settings.cookie_samesite}
    response.set_cookie(ACCESS_COOKIE, access, httponly=True, max_age=settings.access_token_expire_minutes * 60,
                        path="/", **common)
    audit(db, actor.id, "STEP_UP", "user", actor.id, {"methods": methods}, ip, org_id=actor.org_id)
    return {"access_token": access, "step_up_amr": methods, "valid_seconds": 600,
            "user": user_info(db, actor.user, actor.org_id, list(actor.amr))}
