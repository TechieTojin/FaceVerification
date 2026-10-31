"""Passkeys (WebAuthn), passwordless sign-in, live face check after passkey, MFA, recovery and step-up."""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import pyotp
import pytest

from backend.core import crypto
from backend.database import SessionLocal
from backend.models.mfa import AuthChallenge, MfaFactor
from backend.models.user import RoleEnum
from backend.tests.helpers import (
    PASSWORD, admin_headers, default_org_id, drive, enrol_direct, frame_for_step, login, make_user,
)
from backend.tests.webauthn_soft import UV, UP, SoftAuthenticator

API = "/api/v1/auth"


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def step_up_password(client, headers: dict, password: str = PASSWORD) -> dict:
    r = client.post(f"{API}/step-up", json={"method": "password", "password": password}, headers=headers)
    assert r.status_code == 200, r.text
    client.cookies.clear()
    return bearer(r.json()["access_token"])


def register_passkey(client, headers: dict, auth: SoftAuthenticator | None = None, name: str = "Laptop") -> SoftAuthenticator:
    auth = auth or SoftAuthenticator()
    opts = client.post(f"{API}/passkeys/register/options", headers=headers)
    assert opts.status_code == 200, opts.text
    body = opts.json()
    assert body["options"]["authenticatorSelection"]["userVerification"] == "required"
    r = client.post(f"{API}/passkeys/register/verify", headers=headers,
                    json={"challenge_id": body["challenge_id"], "credential": auth.create(body["options"]), "name": name})
    assert r.status_code == 201, r.text
    return auth


def passkey_login(client, auth: SoftAuthenticator, **kw):
    opts = client.post(f"{API}/passkey/options").json()
    return client.post(f"{API}/passkey/verify",
                       json={"challenge_id": opts["challenge_id"], "credential": auth.get(opts["options"], **kw)})


def totp_enable(client, headers: dict) -> tuple[str, list[str]]:
    setup = client.post(f"{API}/totp/setup", headers=headers)
    assert setup.status_code == 200, setup.text
    secret = setup.json()["secret"]
    r = client.post(f"{API}/totp/confirm", headers=headers,
                    json={"factor_id": setup.json()["factor_id"], "code": pyotp.TOTP(secret).now()})
    assert r.status_code == 200, r.text
    return secret, r.json()["recovery_codes"]


def _next_code(secret: str) -> str:
    """A code for the *next* time-step (accepted within the ±1 skew window, newer than the last used)."""
    t = pyotp.TOTP(secret)
    return t.at(time.time() + t.interval)


# ── passkeys ────────────────────────────────────────────────────────────────

def test_passkey_registration_requires_step_up_and_then_signs_in(client):
    user = make_user(RoleEnum.OPERATOR)
    h = login(client, user.email)
    r = client.post(f"{API}/passkeys/register/options", headers=h)
    assert r.status_code == 403 and r.json()["code"] == "STEP_UP_REQUIRED"
    hs = step_up_password(client, h)
    auth = register_passkey(client, hs)
    sec = client.get(f"{API}/security", headers=hs).json()
    assert [p["name"] for p in sec["passkeys"]] == ["Laptop"]

    r = passkey_login(client, auth)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ok" and r.json()["user"]["amr"] == ["webauthn"]
    assert r.json()["user"]["id"] == user.id
    client.cookies.clear()


def test_passkey_rejects_invalid_expired_and_replayed_challenges(client):
    user = make_user()
    auth = register_passkey(client, step_up_password(client, login(client, user.email)))
    opts = client.post(f"{API}/passkey/options").json()
    cred = auth.get(opts["options"])
    r = client.post(f"{API}/passkey/verify", json={"challenge_id": "00000000-0000-0000-0000-000000000000",
                                                   "credential": cred})
    assert r.status_code == 400 and r.json()["code"] == "INVALID_CHALLENGE"

    # A challenge the server did not issue (signed by the real key) fails verification.
    opts = client.post(f"{API}/passkey/options").json()
    forged = auth.get(opts["options"], challenge="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    r = client.post(f"{API}/passkey/verify", json={"challenge_id": opts["challenge_id"], "credential": forged})
    assert r.status_code == 401 and r.json()["code"] == "PASSKEY_INVALID"

    # Expired challenge.
    opts = client.post(f"{API}/passkey/options").json()
    db = SessionLocal()
    db.query(AuthChallenge).filter(AuthChallenge.id == opts["challenge_id"]).update(
        {"expires_at": datetime.utcnow() - timedelta(seconds=1)})
    db.commit()
    db.close()
    r = client.post(f"{API}/passkey/verify", json={"challenge_id": opts["challenge_id"],
                                                   "credential": auth.get(opts["options"])})
    assert r.status_code == 400 and r.json()["code"] == "CHALLENGE_EXPIRED"

    # Replay: the same signed assertion used twice.
    opts = client.post(f"{API}/passkey/options").json()
    cred = auth.get(opts["options"])
    assert client.post(f"{API}/passkey/verify", json={"challenge_id": opts["challenge_id"], "credential": cred}).status_code == 200
    client.cookies.clear()
    r = client.post(f"{API}/passkey/verify", json={"challenge_id": opts["challenge_id"], "credential": cred})
    assert r.status_code == 400 and r.json()["code"] in ("CHALLENGE_USED", "CHALLENGE_EXPIRED")

    # A cloned authenticator whose counter does not advance is rejected.
    opts = client.post(f"{API}/passkey/options").json()
    stale = auth.get(opts["options"], bump=0)
    r = client.post(f"{API}/passkey/verify", json={"challenge_id": opts["challenge_id"], "credential": stale})
    assert r.status_code == 401


def test_passkey_rejects_wrong_origin_missing_uv_and_revoked_credentials(client):
    user = make_user()
    hs = step_up_password(client, login(client, user.email))
    auth = register_passkey(client, hs)
    r = passkey_login(client, auth, origin="https://evil.example")
    assert r.status_code == 401 and r.json()["code"] == "PASSKEY_INVALID"
    r = passkey_login(client, auth, flags=UP)          # user verification not performed
    assert r.status_code == 401

    # Registration from a foreign origin is refused too.
    opts = client.post(f"{API}/passkeys/register/options", headers=hs).json()
    r = client.post(f"{API}/passkeys/register/verify", headers=hs,
                    json={"challenge_id": opts["challenge_id"],
                          "credential": SoftAuthenticator().create(opts["options"], origin="https://evil.example")})
    assert r.status_code == 400 and r.json()["code"] == "REGISTRATION_INVALID"

    cred_id = client.get(f"{API}/security", headers=hs).json()["passkeys"][0]["id"]
    # With a passkey registered, password sign-in needs the second factor...
    assert client.post(f"{API}/login", json={"email": user.email, "password": PASSWORD}).json()["status"] == "mfa_required"
    # ...and password is no longer accepted for step-up.
    r = client.post(f"{API}/step-up", json={"method": "password", "password": PASSWORD}, headers=hs)
    assert r.status_code == 400 and r.json()["code"] == "STRONGER_FACTOR_REQUIRED"
    # A fresh passkey sign-in counts as step-up, so the passkey can be removed.
    r = passkey_login(client, auth)
    client.cookies.clear()
    hp = bearer(r.json()["access_token"])
    assert client.delete(f"{API}/passkeys/{cred_id}", headers=hp).status_code == 200
    r = passkey_login(client, auth)
    assert r.status_code == 401 and r.json()["code"] == "PASSKEY_NOT_RECOGNISED"


def test_passkey_plus_face_when_policy_requires_it(client):
    ha = admin_headers(client)
    assert client.put("/api/v1/policy", json={"changes": {"passkey_login_requires_face": True}},
                      headers=ha).status_code == 200
    user = make_user(RoleEnum.OPERATOR)
    enrol_direct(user.id, identity=7)
    auth = register_passkey(client, step_up_password(client, login(client, user.email)))

    # A pending face token alone cannot be invented.
    assert client.post(f"{API}/login/face/session", json={"pending_token": "x" * 40}).status_code == 401

    r = passkey_login(client, auth)
    assert r.status_code == 200 and r.json()["status"] == "face_required"
    assert "access_token" not in r.json() and not client.cookies.get("fv_access")
    token = r.json()["pending_token"]
    lh = {"X-Login-Token": token}

    # Impostor face → no session.
    s = client.post(f"{API}/login/face/session", json={"pending_token": token}).json()
    final, _ = drive(client, lh, s, identity=9)
    assert final["final_outcome"] == "NOT_MATCHED"
    r = client.post(f"{API}/login/face/complete", json={"pending_token": token})
    assert r.status_code == 401 and r.json()["code"] == "FACE_CHECK_FAILED"
    # The login token cannot touch an operator's session.
    s2 = client.post(f"{API}/login/face/session", json={"pending_token": token}).json()
    op_sess = client.post("/api/v1/sessions", json={"target_user_id": user.id}, headers=ha).json()
    assert client.get(f"/api/v1/sessions/{op_sess['session_id']}", headers=lh).status_code == 404
    final, _ = drive(client, lh, s2, identity=7)
    assert final["final_outcome"] == "VERIFIED"
    r = client.post(f"{API}/login/face/complete", json={"pending_token": token})
    assert r.status_code == 200 and r.json()["user"]["amr"] == ["webauthn", "face"]
    client.cookies.clear()
    assert client.post(f"{API}/login/face/complete", json={"pending_token": token}).status_code == 401  # one use


# ── TOTP / recovery codes / MFA sign-in ─────────────────────────────────────

def test_totp_mfa_sign_in_replay_and_removal(client):
    user = make_user()
    hs = step_up_password(client, login(client, user.email))
    secret, codes = totp_enable(client, hs)
    assert len(codes) == 10

    r = client.post(f"{API}/login", json={"email": user.email, "password": PASSWORD})
    assert r.status_code == 200 and r.json()["status"] == "mfa_required"
    assert "totp" in r.json()["methods"] and "access_token" not in r.json()
    token = r.json()["pending_token"]
    bad = client.post(f"{API}/mfa/verify", json={"pending_token": token, "method": "totp", "code": "000000"})
    assert bad.status_code == 401
    code = _next_code(secret)
    ok = client.post(f"{API}/mfa/verify", json={"pending_token": token, "method": "totp", "code": code})
    assert ok.status_code == 200 and ok.json()["user"]["amr"] == ["pwd", "totp"]
    client.cookies.clear()

    # The same code cannot be used again (time-step replay guard), even in a new sign-in.
    token2 = client.post(f"{API}/login", json={"email": user.email, "password": PASSWORD}).json()["pending_token"]
    r = client.post(f"{API}/mfa/verify", json={"pending_token": token2, "method": "totp", "code": code})
    assert r.status_code == 401
    # Recovery code works exactly once.
    r = client.post(f"{API}/mfa/verify", json={"pending_token": token2, "method": "recovery_code", "code": codes[0]})
    assert r.status_code == 200 and r.json()["user"]["amr"] == ["pwd", "rc"]
    client.cookies.clear()
    token3 = client.post(f"{API}/login", json={"email": user.email, "password": PASSWORD}).json()["pending_token"]
    r = client.post(f"{API}/mfa/verify", json={"pending_token": token3, "method": "recovery_code", "code": codes[0]})
    assert r.status_code == 401

    # Removing the factor needs step-up with the factor itself (password is no longer enough).
    h = bearer(client.post(f"{API}/mfa/verify", json={"pending_token": token3, "method": "recovery_code",
                                                       "code": codes[1]}).json()["access_token"])
    client.cookies.clear()
    factor_id = client.get(f"{API}/security", headers=h).json()["factors"][0]["id"]
    db = SessionLocal()
    db.query(MfaFactor).filter(MfaFactor.id == factor_id).update({"last_used_step": None})
    db.commit()
    db.close()
    r = client.post(f"{API}/step-up", json={"method": "totp", "code": pyotp.TOTP(secret).now()}, headers=h)
    assert r.status_code == 200
    client.cookies.clear()
    hs = bearer(r.json()["access_token"])
    assert client.delete(f"{API}/factors/{factor_id}", headers=hs).status_code == 200
    r = client.post(f"{API}/login", json={"email": user.email, "password": PASSWORD})
    assert r.json()["status"] == "ok"      # back to password-only (recovery codes alone do not force MFA)
    client.cookies.clear()


def test_mfa_attempts_are_limited_and_alerted(client):
    user = make_user()
    totp_enable(client, step_up_password(client, login(client, user.email)))
    token = client.post(f"{API}/login", json={"email": user.email, "password": PASSWORD}).json()["pending_token"]
    for _ in range(5):
        assert client.post(f"{API}/mfa/verify", json={"pending_token": token, "method": "totp",
                                                       "code": "123456"}).status_code == 401
    r = client.post(f"{API}/mfa/verify", json={"pending_token": token, "method": "totp", "code": "123456"})
    assert r.status_code == 429
    alerts = client.get("/api/v1/security/alerts", headers=admin_headers(client)).json()["alerts"]
    assert any(a["rule"] == "MFA_FAILURES" for a in alerts)


def test_policy_can_require_mfa_for_admins_with_enrolment_during_sign_in(client):
    ha = admin_headers(client)
    assert client.put("/api/v1/policy", json={"changes": {"mfa_required_for_admins": True}}, headers=ha).status_code == 200
    admin2 = make_user(RoleEnum.ADMIN)
    r = client.post(f"{API}/login", json={"email": admin2.email, "password": PASSWORD})
    assert r.json()["status"] == "mfa_required" and r.json()["enrol_required"] is True
    token = r.json()["pending_token"]
    setup = client.post(f"{API}/mfa/enrol/totp", json={"pending_token": token}).json()
    r = client.post(f"{API}/mfa/enrol/totp/confirm", json={"pending_token": token, "factor_id": setup["factor_id"],
                                                            "code": pyotp.TOTP(setup["secret"]).now()})
    assert r.status_code == 200 and r.json()["user"]["amr"] == ["pwd", "totp"] and len(r.json()["recovery_codes"]) == 10
    client.cookies.clear()
    # Operators are unaffected by the admin-only rule.
    op = make_user(RoleEnum.OPERATOR)
    assert client.post(f"{API}/login", json={"email": op.email, "password": PASSWORD}).json()["status"] == "ok"
    client.cookies.clear()


def test_passkey_as_second_factor(client):
    user = make_user()
    hs = step_up_password(client, login(client, user.email))
    auth = register_passkey(client, hs)
    r = client.post(f"{API}/login", json={"email": user.email, "password": PASSWORD})
    assert r.json()["status"] == "mfa_required" and "webauthn" in r.json()["methods"]
    token = r.json()["pending_token"]
    opts = client.post(f"{API}/mfa/webauthn/options", json={"pending_token": token}).json()
    r = client.post(f"{API}/mfa/verify", json={"pending_token": token, "method": "webauthn",
                                               "challenge_id": opts["challenge_id"],
                                               "credential": auth.get(opts["options"])})
    assert r.status_code == 200 and r.json()["user"]["amr"] == ["pwd", "webauthn"]
    client.cookies.clear()


def test_email_otp_is_unavailable_without_smtp(client):
    user = make_user()
    totp_enable(client, step_up_password(client, login(client, user.email)))
    r = client.post(f"{API}/login", json={"email": user.email, "password": PASSWORD})
    assert "email_otp" not in r.json()["methods"]
    r = client.post(f"{API}/mfa/email-otp", json={"pending_token": r.json()["pending_token"]})
    assert r.status_code == 503 and r.json()["code"] == "EMAIL_UNAVAILABLE"


def test_high_risk_actions_need_face_and_factor_when_policy_requires(client):
    ha = admin_headers(client)
    assert client.put("/api/v1/policy", json={"changes": {"high_risk_requires_face": True}}, headers=ha).status_code == 200
    from backend.tests.helpers import admin_id
    enrol_direct(admin_id(), identity=11)
    from backend.config import settings
    hs = step_up_password(client, ha, settings.initial_admin_password)
    r = client.post("/api/v1/organizations", json={"name": "Needs Face"}, headers=hs)
    assert r.status_code == 403 and r.json()["step_up_level"] == "face_and_factor"

    start = client.post(f"{API}/step-up/face/session", headers=ha)
    assert start.status_code == 201, start.text
    face_token, sess = start.json()["face_token"], start.json()["session"]
    final, _ = drive(client, {"X-Login-Token": face_token}, sess, identity=11)
    assert final["final_outcome"] == "VERIFIED"
    r = client.post(f"{API}/step-up", json={"method": "password", "password": settings.initial_admin_password,
                                            "face_token": face_token}, headers=ha)
    assert r.status_code == 200 and r.json()["step_up_amr"] == ["face", "pwd"]
    client.cookies.clear()
    hf = bearer(r.json()["access_token"])
    r = client.post("/api/v1/organizations", json={"name": "Needs Face"}, headers=hf)
    assert r.status_code == 201, r.text
    # The face check cannot be completed into a sign-in.
    assert client.post(f"{API}/login/face/complete", json={"pending_token": face_token}).status_code == 401


def test_admin_assisted_recovery_resets_factors_of_exclusive_members_only(client):
    from backend.tests.helpers import add_member, make_org
    user = make_user(RoleEnum.OPERATOR, name="Lost Phone")
    secret, _ = totp_enable(client, step_up_password(client, login(client, user.email)))
    ha = admin_headers(client)
    r = client.post(f"/api/v1/users/{user.id}/reset-mfa", headers=ha)
    assert r.status_code == 403 and r.json()["code"] == "STEP_UP_REQUIRED"
    from backend.config import settings
    hs = step_up_password(client, ha, settings.initial_admin_password)
    r = client.post(f"/api/v1/users/{user.id}/reset-mfa", headers=hs)
    assert r.status_code == 200 and r.json()["removed"]["factors"] == 1
    # Password sign-in works again without a second factor; the user was notified.
    r = client.post(f"{API}/login", json={"email": user.email, "password": PASSWORD})
    assert r.json()["status"] == "ok"
    client.cookies.clear()
    notes = client.get("/api/v1/notifications", headers=login(client, user.email)).json()["notifications"]
    assert any(n["title"] == "Sign-in factors reset" for n in notes)
    # A shared account (member elsewhere too) cannot be reset by one organisation.
    shared = make_user(RoleEnum.OPERATOR)
    add_member(shared.id, make_org("Other", "other-reset"))
    assert client.post(f"/api/v1/users/{shared.id}/reset-mfa", headers=hs).status_code == 409
