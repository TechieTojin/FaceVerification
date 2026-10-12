"""Tests for user management endpoints."""
from __future__ import annotations
import pytest


def test_list_users_admin(client, auth_headers):
    resp = client.get("/api/v1/users", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "users" in data
    assert data["total"] >= 1


def test_list_users_unauthenticated(client):
    resp = client.get("/api/v1/users")
    assert resp.status_code == 401


def test_create_user(client, auth_headers):
    resp = client.post("/api/v1/users", json={
        "email": "test@example.com",
        "full_name": "Test User",
        "password": "TestPass123!",
        "role": "OPERATOR",
    }, headers=auth_headers)
    assert resp.status_code == 201
    data = resp.json()
    assert data["email"] == "test@example.com"
    assert data["role"] == "OPERATOR"


def test_create_duplicate_user(client, auth_headers):
    client.post("/api/v1/users", json={
        "email": "dup@example.com",
        "full_name": "Dup",
        "password": "TestPass123!",
        "role": "OPERATOR",
    }, headers=auth_headers)
    resp = client.post("/api/v1/users", json={
        "email": "dup@example.com",
        "full_name": "Dup2",
        "password": "TestPass123!",
        "role": "OPERATOR",
    }, headers=auth_headers)
    assert resp.status_code == 409


def test_get_user(client, auth_headers):
    create = client.post("/api/v1/users", json={
        "email": "getuser@example.com",
        "full_name": "Get User",
        "password": "TestPass123!",
    }, headers=auth_headers)
    user_id = create.json()["id"]
    resp = client.get(f"/api/v1/users/{user_id}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == user_id


def test_update_user(client, auth_headers):
    create = client.post("/api/v1/users", json={
        "email": "updateuser@example.com",
        "full_name": "Update User",
        "password": "TestPass123!",
    }, headers=auth_headers)
    user_id = create.json()["id"]
    resp = client.put(f"/api/v1/users/{user_id}", json={"full_name": "Updated Name"}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["full_name"] == "Updated Name"


def test_disable_user(client, auth_headers):
    create = client.post("/api/v1/users", json={
        "email": "disableuser@example.com",
        "full_name": "Disable User",
        "password": "TestPass123!",
    }, headers=auth_headers)
    user_id = create.json()["id"]
    resp = client.post(f"/api/v1/users/{user_id}/disable", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["is_active"] == False
