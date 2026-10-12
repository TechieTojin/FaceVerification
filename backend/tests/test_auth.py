"""Core authentication endpoint tests."""
from __future__ import annotations
import pytest
from backend.config import settings


def test_login_success(client):
    resp = client.post("/api/v1/auth/login", json={
        "email": settings.initial_admin_email,
        "password": settings.initial_admin_password,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "user" in data
    assert data["user"]["email"] == settings.initial_admin_email


def test_login_wrong_password(client):
    resp = client.post("/api/v1/auth/login", json={
        "email": settings.initial_admin_email,
        "password": "wrongpassword",
    })
    assert resp.status_code == 401


def test_login_unknown_user(client):
    resp = client.post("/api/v1/auth/login", json={
        "email": "noone@example.com",
        "password": "password",
    })
    assert resp.status_code == 401


def test_login_sets_cookies(client):
    resp = client.post("/api/v1/auth/login", json={
        "email": settings.initial_admin_email,
        "password": settings.initial_admin_password,
    })
    assert resp.status_code == 200
    assert "fv_access" in client.cookies
    assert "fv_refresh" in client.cookies
    assert "fv_csrf" in client.cookies


def test_me_authenticated_bearer(client, auth_headers):
    resp = client.get("/api/v1/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == settings.initial_admin_email
    assert data["role"] == "ADMIN"


def test_me_authenticated_cookie(logged_in_client):
    resp = logged_in_client.get("/api/v1/auth/me")
    assert resp.status_code == 200
    assert resp.json()["email"] == settings.initial_admin_email


def test_me_unauthenticated(client):
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 401


def test_logout_bearer(client, auth_headers):
    resp = client.post("/api/v1/auth/logout", headers=auth_headers)
    assert resp.status_code == 200


def test_logout_clears_cookies(logged_in_client, csrf_headers):
    resp = logged_in_client.post("/api/v1/auth/logout", headers=csrf_headers)
    assert resp.status_code == 200
    # After logout the access cookie should be gone
    assert logged_in_client.cookies.get("fv_access") in (None, "")


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
    assert resp.json()["version"] == "2.1.0"
