"""v2.1 security hardening tests.

Covers: rate limiting, account lockout, CSRF, refresh token rotation,
        revoked token reuse detection, logout, session restoration,
        RBAC, password change.
"""
from __future__ import annotations
import pytest
from unittest.mock import patch
from backend.config import settings
from backend.models.auth import LoginAttempt
from backend.services.auth_service import record_login_attempt, hash_password


# ── Helpers ──────────────────────────────────────────────────────────────────

def _login(client, email=None, password=None):
    return client.post("/api/v1/auth/login", json={
        "email": email or settings.initial_admin_email,
        "password": password or settings.initial_admin_password,
    })


def _seed_failures(db_session, email, ip, count):
    """Insert `count` consecutive failed login attempts directly into DB."""
    for _ in range(count):
        record_login_attempt(db_session, email, ip, False)


# ── Rate limiting ─────────────────────────────────────────────────────────────

def test_account_lockout_after_consecutive_failures(client, db_session):
    """After N consecutive failures, the account is locked."""
    target = "lockout_test@example.com"
    ip = "testclient"

    _seed_failures(db_session, target, ip, settings.login_lockout_threshold)

    resp = _login(client, email=target, password="whatever")
    assert resp.status_code == 429
    data = resp.json()
    assert "locked" in data["detail"].lower()
    assert "Retry-After" in resp.headers


def test_lockout_resets_after_success(client, db_session):
    """A successful login clears the consecutive-failure counter."""
    target = "lockout_reset@example.com"
    ip = "testclient"

    # Seed failures below threshold
    _seed_failures(db_session, target, ip, settings.login_lockout_threshold - 1)
    # Record a success
    record_login_attempt(db_session, target, ip, True)

    # Now another failure — should NOT be locked (counter reset by the success)
    resp = _login(client, email=target, password="wrong")
    # 401 = wrong password (not locked), which is correct
    assert resp.status_code == 401


def test_ip_rate_limit(client, db_session):
    """After per-IP limit is exceeded, all logins from that IP are blocked."""
    ip = "testclient"
    # Flood attempts from this IP
    for i in range(settings.login_rate_limit_per_ip):
        record_login_attempt(db_session, f"flood{i}@x.com", ip, False)

    resp = _login(client)
    assert resp.status_code == 429
    assert "Too many" in resp.json()["detail"] or "locked" in resp.json()["detail"]


def test_account_rate_limit(client, db_session):
    """After per-account limit is exceeded (all attempts), account is blocked."""
    target = "acct_rate@example.com"
    ip = "testclient"
    for i in range(settings.login_rate_limit_per_account):
        record_login_attempt(db_session, target, ip, False)

    resp = _login(client, email=target, password="wrong")
    assert resp.status_code == 429


# ── CSRF ──────────────────────────────────────────────────────────────────────

def test_csrf_rejected_without_header(logged_in_client):
    """State-changing request via cookie auth without X-CSRF-Token → 403."""
    resp = logged_in_client.post("/api/v1/auth/logout")
    assert resp.status_code == 403
    assert "CSRF" in resp.json()["detail"]


def test_csrf_accepted_with_correct_header(logged_in_client, csrf_headers):
    """Same request with correct X-CSRF-Token header → succeeds."""
    resp = logged_in_client.post("/api/v1/auth/logout", headers=csrf_headers)
    assert resp.status_code == 200


def test_csrf_rejected_with_wrong_token(logged_in_client):
    """Wrong CSRF token → 403."""
    resp = logged_in_client.post(
        "/api/v1/auth/logout",
        headers={"X-CSRF-Token": "completely-wrong-token"},
    )
    assert resp.status_code == 403


def test_csrf_not_required_for_bearer_auth(client, auth_headers):
    """Bearer-token requests are not subject to CSRF (no cookie used)."""
    resp = client.post("/api/v1/auth/logout", headers=auth_headers)
    assert resp.status_code == 200


def test_csrf_not_required_for_safe_methods(logged_in_client):
    """GET requests are never CSRF-checked."""
    resp = logged_in_client.get("/api/v1/auth/me")
    assert resp.status_code == 200


def test_csrf_exempt_login(client):
    """Login endpoint is CSRF-exempt (no session cookie exists yet)."""
    resp = _login(client)
    assert resp.status_code == 200


def test_csrf_exempt_refresh(logged_in_client):
    """Refresh endpoint is CSRF-exempt (authenticated by HttpOnly refresh cookie)."""
    resp = logged_in_client.post("/api/v1/auth/refresh")
    assert resp.status_code == 200


# ── Refresh token rotation ────────────────────────────────────────────────────

def test_refresh_issues_new_access_token(logged_in_client):
    """POST /auth/refresh returns new access cookie."""
    resp = logged_in_client.post("/api/v1/auth/refresh")
    assert resp.status_code == 200
    assert "expires_in" in resp.json()
    assert "fv_access" in logged_in_client.cookies


def test_refresh_rotates_refresh_token(logged_in_client):
    """The refresh token cookie changes after each refresh call."""
    old_refresh = logged_in_client.cookies.get("fv_refresh")
    logged_in_client.post("/api/v1/auth/refresh")
    new_refresh = logged_in_client.cookies.get("fv_refresh")
    assert old_refresh != new_refresh


def test_revoked_refresh_token_rejected(logged_in_client):
    """Using the old refresh token after rotation → 401 (token reuse detection)."""
    old_refresh = logged_in_client.cookies.get("fv_refresh")

    # Rotate once
    logged_in_client.post("/api/v1/auth/refresh")

    # Re-inject the old refresh token and try to use it
    logged_in_client.cookies.set("fv_refresh", old_refresh)
    resp = logged_in_client.post("/api/v1/auth/refresh")
    assert resp.status_code == 401


def test_refresh_without_cookie_rejected(client):
    """Refresh with no cookie → 401."""
    resp = client.post("/api/v1/auth/refresh")
    assert resp.status_code == 401


# ── Logout and session invalidation ──────────────────────────────────────────

def test_logout_prevents_refresh(logged_in_client, csrf_headers):
    """After logout, the refresh token is revoked and cannot mint new tokens."""
    old_refresh = logged_in_client.cookies.get("fv_refresh")

    logged_in_client.post("/api/v1/auth/logout", headers=csrf_headers)

    # Re-inject old refresh cookie and try to refresh
    logged_in_client.cookies.set("fv_refresh", old_refresh)
    resp = logged_in_client.post("/api/v1/auth/refresh")
    assert resp.status_code == 401


def test_logout_clears_access_cookie(logged_in_client, csrf_headers):
    logged_in_client.post("/api/v1/auth/logout", headers=csrf_headers)
    assert logged_in_client.cookies.get("fv_access") in (None, "")


# ── Session restoration ───────────────────────────────────────────────────────

def test_me_works_with_cookie_after_login(logged_in_client):
    """Simulates browser reload: /me via cookie restores the session."""
    resp = logged_in_client.get("/api/v1/auth/me")
    assert resp.status_code == 200
    assert resp.json()["role"] == "ADMIN"


def test_me_fails_after_logout(logged_in_client, csrf_headers):
    logged_in_client.post("/api/v1/auth/logout", headers=csrf_headers)
    # Clear access cookie to simulate browser state after logout
    logged_in_client.cookies.delete("fv_access")
    resp = logged_in_client.get("/api/v1/auth/me")
    assert resp.status_code == 401


# ── Role-based access control ─────────────────────────────────────────────────

def test_admin_can_list_users(client, auth_headers):
    resp = client.get("/api/v1/users", headers=auth_headers)
    assert resp.status_code == 200


def test_operator_cannot_list_users(client, db_session):
    """An OPERATOR-role user cannot access admin-only endpoints."""
    from backend.services.auth_service import create_user
    from backend.models.user import RoleEnum
    op = create_user(db_session, "operator@rbac.test", "Op", "Passw0rd!!!", RoleEnum.OPERATOR)
    db_session.commit()

    resp = client.post("/api/v1/auth/login", json={
        "email": "operator@rbac.test",
        "password": "Passw0rd!!!",
    })
    assert resp.status_code == 200
    op_token = resp.json()["access_token"]

    resp2 = client.get("/api/v1/users", headers={"Authorization": f"Bearer {op_token}"})
    assert resp2.status_code == 403


# ── Password change ───────────────────────────────────────────────────────────

def test_password_change_success(client, db_session):
    """Change password; old password rejected, new password accepted."""
    from backend.services.auth_service import create_user
    email = "pwchange@example.com"
    old_pw = "OldPass123!"
    new_pw = "NewPass456!"
    create_user(db_session, email, "PW Change", old_pw)
    db_session.commit()

    # Login
    login_resp = client.post("/api/v1/auth/login", json={"email": email, "password": old_pw})
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    csrf = client.cookies.get("fv_csrf", "")

    # Change password (via cookie + CSRF since we're cookie-authenticated)
    resp = client.post(
        "/api/v1/auth/change-password",
        json={"current_password": old_pw, "new_password": new_pw},
        headers={"X-CSRF-Token": csrf},
    )
    assert resp.status_code == 200

    # Old password rejected
    bad = client.post("/api/v1/auth/login", json={"email": email, "password": old_pw})
    assert bad.status_code == 401

    # New password accepted
    good = client.post("/api/v1/auth/login", json={"email": email, "password": new_pw})
    assert good.status_code == 200


def test_password_change_wrong_current(client, db_session, auth_headers):
    """Wrong current password → 400."""
    resp = client.post(
        "/api/v1/auth/change-password",
        json={"current_password": "definitely-wrong", "new_password": "NewPass789!"},
        headers=auth_headers,
    )
    assert resp.status_code == 400
    assert "incorrect" in resp.json()["detail"].lower()


def test_password_change_revokes_refresh_tokens(client, db_session):
    """Password change revokes all refresh tokens, forcing re-login."""
    from backend.services.auth_service import create_user
    email = "pwrevoke@example.com"
    old_pw = "OldRevoke1!"
    new_pw = "NewRevoke2!"
    create_user(db_session, email, "PW Revoke", old_pw)
    db_session.commit()

    # Login (gets refresh cookie)
    login_resp = client.post("/api/v1/auth/login", json={"email": email, "password": old_pw})
    old_refresh = client.cookies.get("fv_refresh")
    csrf = client.cookies.get("fv_csrf", "")

    # Change password
    client.post(
        "/api/v1/auth/change-password",
        json={"current_password": old_pw, "new_password": new_pw},
        headers={"X-CSRF-Token": csrf},
    )

    # Old refresh token should be revoked
    client.cookies.set("fv_refresh", old_refresh)
    resp = client.post("/api/v1/auth/refresh")
    assert resp.status_code == 401


# ── Settings validation ───────────────────────────────────────────────────────

def test_settings_floor_validation(client, auth_headers):
    """session_timeout_seconds below floor (30) should be rejected."""
    resp = client.put(
        "/api/v1/settings",
        json={"session_timeout_seconds": "5"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_settings_valid_update(client, auth_headers):
    resp = client.put(
        "/api/v1/settings",
        json={"session_timeout_seconds": "60"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert "session_timeout_seconds" in resp.json()["updated"]


def test_settings_unknown_key_rejected(client, auth_headers):
    resp = client.put(
        "/api/v1/settings",
        json={"evil_key": "value"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_enable_liveness_cannot_be_disabled_silently(client, auth_headers):
    """enable_liveness must remain a valid bool — any other value rejected."""
    resp = client.put(
        "/api/v1/settings",
        json={"enable_liveness": "maybe"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


# ── Embedding encryption key validation ──────────────────────────────────────

def test_fernet_key_missing_raises():
    """get_fernet_key() raises RuntimeError when EMBEDDING_ENCRYPTION_KEY is unset."""
    from backend.config import Settings
    s = Settings.model_construct(
        secret_key="test-only-secret-key-not-for-production-32ch!!",
        embedding_encryption_key="",
    )
    with pytest.raises(RuntimeError, match="EMBEDDING_ENCRYPTION_KEY is not set"):
        s.get_fernet_key()


def test_fernet_key_invalid_length_raises():
    """get_fernet_key() raises RuntimeError when key length is not 44 chars."""
    from backend.config import Settings
    s = Settings.model_construct(
        secret_key="test-only-secret-key-not-for-production-32ch!!",
        embedding_encryption_key="tooshort",
    )
    with pytest.raises(RuntimeError, match="invalid length"):
        s.get_fernet_key()


def test_fernet_key_valid_returns_bytes():
    """get_fernet_key() returns the raw bytes of a valid 44-char key."""
    from cryptography.fernet import Fernet
    from backend.config import Settings
    valid = Fernet.generate_key().decode()
    s = Settings.model_construct(
        secret_key="test-only-secret-key-not-for-production-32ch!!",
        embedding_encryption_key=valid,
    )
    result = s.get_fernet_key()
    assert result == valid.encode()
    assert len(result) == 44
