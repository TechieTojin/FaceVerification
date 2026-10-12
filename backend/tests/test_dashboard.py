"""Tests for dashboard and analytics endpoints."""
from __future__ import annotations
import pytest


def test_dashboard_summary(client, auth_headers):
    resp = client.get("/api/v1/dashboard/summary", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "total_users" in data
    assert "active_enrolments" in data
    assert "total_verifications" in data
    assert "success_rate" in data
    assert "backend_healthy" in data
    assert data["backend_healthy"] == True


def test_analytics(client, auth_headers):
    resp = client.get("/api/v1/analytics", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "verifications_by_day" in data
    assert "outcome_distribution" in data
    assert "latency_histogram" in data


def test_security_events(client, auth_headers):
    resp = client.get("/api/v1/security/events", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "events" in data
    assert "total" in data


def test_settings_get(client, auth_headers):
    resp = client.get("/api/v1/settings", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "enable_liveness" in data


def test_settings_update(client, auth_headers):
    resp = client.put("/api/v1/settings", json={"max_face_distance": "0.70"}, headers=auth_headers)
    assert resp.status_code == 200
