"""Authentication: cookies, CSRF, refresh rotation and reuse, lockout, password changes, device sessions."""
from __future__ import annotations

import time

from backend.config import settings
from backend.models.user import RoleEnum
from backend.services.auth_service import record_login_attempt
from backend.database import SessionLocal
from backend.tests.helpers import PASSWORD, admin_headers, login, make_user

A = {"email": settings.initial_admin_email, "password": settings.initial_admin_password}


def cookie_login(client):
    r = client.post("/api/v1/auth/login", json=A)
    assert r.status_code == 200
    return {"X-CSRF-Token": client.cookies.get("fv_csrf")}


def test_login_sets_httponly_cookies_and_returns_permissions(client):
    r = client.post("/api/v1/auth/login", json=A)
    assert r.status_code == 200
    body = r.json()
    assert body["user"]["role"] == "ADMIN" and "policy:manage" in body["user"]["permissions"]
    raw = r.headers.get_list("set-cookie")
    assert any(c.startswith("fv_access=") and "HttpOnly" in c for c in raw)
    assert any(c.startswith("fv_refresh=") and "HttpOnly" in c and "Path=/api/v1/auth" in c for c in raw)
    assert any(c.startswith("fv_csrf=") and "HttpOnly" not in c for c in raw)
    assert "hashed_password" not in r.text


def test_login_failures_are_uniform(client):
    r1 = client.post("/api/v1/auth/login", json={"email": A["email"], "password": "wrong"})
    r2 = client.post("/api/v1/auth/login", json={"email": "nobody@test.local", "password": "wrong"})
    assert r1.status_code == r2.status_code == 401
    assert r1.json() == r2.json()


def test_unknown_email_still_costs_a_hash(client, monkeypatch):
    from backend.services import auth_service
    calls = []
    real = auth_service.verify_password
    monkeypatch.setattr(auth_service, "verify_password", lambda p, h: calls.append(h) or real(p, h))
    client.post("/api/v1/auth/login", json={"email": "ghost@test.local", "password": "x"})
    assert calls == [auth_service._DUMMY_HASH]


def test_disabled_user_cannot_log_in(client):
    u = make_user(RoleEnum.OPERATOR, active=False)
    assert client.post("/api/v1/auth/login", json={"email": u.email, "password": PASSWORD}).status_code == 401


def test_me_via_cookie_and_bearer(client):
    cookie_login(client)
    assert client.get("/api/v1/auth/me").status_code == 200
    client.cookies.clear()
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/auth/me", headers=admin_headers(client)).json()["email"] == A["email"]


def test_csrf_required_for_cookie_auth(client):
    h = cookie_login(client)
    r = client.post("/api/v1/auth/logout")
    assert r.status_code == 403 and r.json()["code"] == "CSRF_FAILED"
    assert client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": "forged"}).status_code == 403
    assert client.post("/api/v1/auth/logout", headers=h).status_code == 200


def test_csrf_not_applied_to_bearer_or_safe_methods(client):
    h = admin_headers(client)
    assert client.post("/api/v1/auth/logout", headers=h).status_code == 200
    cookie_login(client)
    assert client.get("/api/v1/dashboard/summary").status_code == 200


def test_refresh_rotates_and_reuse_revokes_family(client):
    cookie_login(client)
    old = client.cookies.get("fv_refresh")
    r = client.post("/api/v1/auth/refresh")
    assert r.status_code == 200 and client.cookies.get("fv_refresh") != old
    new = client.cookies.get("fv_refresh")
    client.cookies.clear()
    client.cookies.set("fv_refresh", old)
    assert client.post("/api/v1/auth/refresh").status_code == 401
    client.cookies.clear()
    client.cookies.set("fv_refresh", new)
    assert client.post("/api/v1/auth/refresh").status_code == 401   # whole family revoked
    alerts = client.get("/api/v1/security/alerts", headers=admin_headers(client)).json()["alerts"]
    assert any(a["rule"] == "REFRESH_TOKEN_REUSE" for a in alerts)


def test_logout_revokes_refresh(client):
    h = cookie_login(client)
    old = client.cookies.get("fv_refresh")
    client.post("/api/v1/auth/logout", headers=h)
    client.cookies.clear()
    client.cookies.set("fv_refresh", old)
    assert client.post("/api/v1/auth/refresh").status_code == 401


def test_lockout_after_consecutive_failures_raises_alert(client):
    for _ in range(settings.login_lockout_threshold):
        client.post("/api/v1/auth/login", json={"email": A["email"], "password": "bad"})
    r = client.post("/api/v1/auth/login", json=A)
    assert r.status_code == 429 and "locked" in r.json()["detail"].lower() and "Retry-After" in r.headers
    db = SessionLocal()
    from backend.models.verification import SecurityAlert
    assert db.query(SecurityAlert).filter(SecurityAlert.rule == "ACCOUNT_LOCKOUT").count() == 1
    db.close()


def test_ip_rate_limit(client):
    db = SessionLocal()
    for i in range(settings.login_rate_limit_per_ip):
        record_login_attempt(db, f"flood{i}@x.com", "testclient", False)
    db.close()
    assert client.post("/api/v1/auth/login", json=A).status_code == 429


def test_password_change_policy_and_token_invalidation(client):
    u = make_user(RoleEnum.OPERATOR)
    h = login(client, u.email)
    r = client.post("/api/v1/auth/change-password", json={"current_password": PASSWORD, "new_password": "short"}, headers=h)
    assert r.status_code == 422
    r = client.post("/api/v1/auth/change-password", json={"current_password": "wrong", "new_password": "N3w-Passw0rd!!"},
                    headers=h)
    assert r.status_code == 400
    time.sleep(1.1)
    r = client.post("/api/v1/auth/change-password", json={"current_password": PASSWORD, "new_password": "N3w-Passw0rd!!"},
                    headers=h)
    assert r.status_code == 200
    assert client.get("/api/v1/auth/me", headers=h).status_code == 401       # old access token invalid
    assert client.post("/api/v1/auth/login", json={"email": u.email, "password": PASSWORD}).status_code == 401
    assert client.post("/api/v1/auth/login", json={"email": u.email, "password": "N3w-Passw0rd!!"}).status_code == 200


def test_device_sessions_list_and_revoke(client):
    cookie_login(client)
    h2 = admin_headers(client)                 # a second sign-in (bearer)
    cookie_login(client)
    s = client.get("/api/v1/auth/sessions").json()["sessions"]
    assert len(s) >= 2 and sum(1 for x in s if x["current"]) == 1
    csrf = {"X-CSRF-Token": client.cookies.get("fv_csrf")}
    n = client.post("/api/v1/auth/sessions/revoke-others", headers=csrf).json()["revoked"]
    assert n >= 1
    assert len(client.get("/api/v1/auth/sessions").json()["sessions"]) == 1
    assert h2
