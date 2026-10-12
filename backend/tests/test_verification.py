"""Tests for verification session endpoints."""
from __future__ import annotations
import io
import pytest
from unittest.mock import patch
import numpy as np


def _create_enrolled_user(client, auth_headers, email):
    """Create a user and mock-enrol them."""
    create = client.post("/api/v1/users", json={
        "email": email,
        "full_name": "Verify Test",
        "password": "TestPass123!",
    }, headers=auth_headers)
    user_id = create.json()["id"]
    client.post(f"/api/v1/enrolments/{user_id}/consent", json={"consent_given": True}, headers=auth_headers)
    return user_id


def test_create_session_not_enrolled(client, auth_headers):
    create = client.post("/api/v1/users", json={
        "email": "notenrolledsession@example.com",
        "full_name": "Not Enrolled",
        "password": "TestPass123!",
    }, headers=auth_headers)
    user_id = create.json()["id"]
    resp = client.post("/api/v1/verification/sessions", json={"target_user_id": user_id}, headers=auth_headers)
    assert resp.status_code == 400


def test_session_not_found(client, auth_headers):
    resp = client.get("/api/v1/verification/sessions/nonexistent-id", headers=auth_headers)
    assert resp.status_code == 404


def test_verification_events_empty(client, auth_headers):
    resp = client.get("/api/v1/verification/events", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "events" in data
    assert "total" in data
