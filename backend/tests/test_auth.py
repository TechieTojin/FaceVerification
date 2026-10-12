"""Tests for authentication endpoints."""
from __future__ import annotations
import pytest


def test_login_success(client):
    resp = client.post("/api/v1/auth/login", json={
        "email": "admin@faceverification.local",
        "password": "Admin1234!",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password(client):
    resp = client.post("/api/v1/auth/login", json={
        "email": "admin@faceverification.local",
        "password": "wrongpassword",
    })
    assert resp.status_code == 401


def test_login_unknown_user(client):
    resp = client.post("/api/v1/auth/login", json={
        "email": "noone@example.com",
        "password": "password",
    })
    assert resp.status_code == 401


def test_me_authenticated(client, auth_headers):
    resp = client.get("/api/v1/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == "admin@faceverification.local"
    assert data["role"] == "ADMIN"


def test_me_unauthenticated(client):
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 401


def test_logout(client, auth_headers):
    resp = client.post("/api/v1/auth/logout", headers=auth_headers)
    assert resp.status_code == 200


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
