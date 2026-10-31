"""User management and the role/permission matrix."""
from __future__ import annotations

import pytest

from backend.models.user import RoleEnum
from backend.tests.helpers import admin_headers, admin_id, enrol_direct, login, make_user, start_verify, drive


def test_create_validate_and_list_users(client):
    h = admin_headers(client)
    r = client.post("/api/v1/users", json={"email": "Alice@Example.COM", "full_name": "Alice", "password": "Sup3r-Secret!",
                                           "role": "OPERATOR"}, headers=h)
    assert r.status_code == 201 and r.json()["email"] == "alice@example.com"
    assert client.post("/api/v1/users", json={"email": "alice@example.com", "full_name": "A", "password": "Sup3r-Secret!"},
                       headers=h).status_code == 409
    assert client.post("/api/v1/users", json={"email": "bad", "full_name": "A", "password": "Sup3r-Secret!"},
                       headers=h).status_code == 422
    assert client.post("/api/v1/users", json={"email": "b@example.com", "full_name": "B", "password": "weakpass"},
                       headers=h).status_code == 422
    data = client.get("/api/v1/users?search=alice&role=OPERATOR&status=active", headers=h).json()
    assert data["total"] == 1 and data["users"][0]["enrolment_status"] == "NOT_ENROLLED"


def test_pagination_and_enrolment_filter(client):
    h = admin_headers(client)
    for _ in range(7):
        make_user(RoleEnum.OPERATOR)
    enrolled = make_user(RoleEnum.OPERATOR)
    enrol_direct(enrolled.id)
    p = client.get("/api/v1/users?page=2&page_size=5", headers=h).json()
    assert p["total"] == 9 and len(p["users"]) == 4
    e = client.get("/api/v1/users?enrolment=ACTIVE", headers=h).json()
    assert [u["id"] for u in e["users"]] == [enrolled.id]


def test_edit_role_disable_enable_reset(client):
    h = admin_headers(client)
    u = make_user(RoleEnum.OPERATOR)
    assert client.patch(f"/api/v1/users/{u.id}", json={"role": "VIEWER", "full_name": "Renamed"}, headers=h).json()["role"] == "VIEWER"
    assert client.post(f"/api/v1/users/{u.id}/disable", headers=h).json()["is_active"] is False
    assert client.post("/api/v1/auth/login", json={"email": u.email, "password": "Str0ng-Test-Pass!"}).status_code == 401
    assert client.post(f"/api/v1/users/{u.id}/enable", headers=h).json()["is_active"] is True
    assert client.post(f"/api/v1/users/{u.id}/reset-password", json={"new_password": "Reset-Pass-123!"}, headers=h).status_code == 200
    assert client.post("/api/v1/auth/login", json={"email": u.email, "password": "Reset-Pass-123!"}).status_code == 200
    client.cookies.clear()
    actions = [e["action"] for e in client.get(f"/api/v1/users/{u.id}/audit", headers=h).json()["entries"]]
    assert {"USER_UPDATED", "USER_DISABLED", "USER_ENABLED", "PASSWORD_RESET"} <= set(actions)


def test_last_admin_and_self_protection(client):
    h = admin_headers(client)
    aid = admin_id()
    assert client.post(f"/api/v1/users/{aid}/disable", headers=h).status_code == 409
    assert client.patch(f"/api/v1/users/{aid}", json={"role": "OPERATOR"}, headers=h).status_code == 409
    assert client.delete(f"/api/v1/users/{aid}", headers=h).status_code == 409


def test_delete_refused_when_history_exists(client):
    h = admin_headers(client)
    fresh = make_user(RoleEnum.OPERATOR)
    assert client.delete(f"/api/v1/users/{fresh.id}", headers=h).status_code == 204
    target = make_user(RoleEnum.OPERATOR)
    enrol_direct(target.id)
    start_verify(client, h, target.id)
    r = client.delete(f"/api/v1/users/{target.id}", headers=h)
    assert r.status_code == 409 and "Disable" in r.json()["detail"]


def test_directory_is_minimal_and_filters_enrolled(client):
    op = make_user(RoleEnum.OPERATOR)
    t = make_user(RoleEnum.OPERATOR, name="Zed Enrolled")
    enrol_direct(t.id)
    h = login(client, op.email)
    d = client.get("/api/v1/users/directory?enrolled_only=true", headers=h).json()["users"]
    assert [x["id"] for x in d] == [t.id]
    assert set(d[0]) == {"id", "email", "full_name", "role", "is_active", "enrolment_status", "enrolment_usable",
                         "reenrolment_requested"}


# Endpoint × role matrix: (method, path, body) → expected allowed roles.
MATRIX = [
    ("GET", "/api/v1/dashboard/summary", None, {"ADMIN", "OPERATOR", "VIEWER"}),
    ("GET", "/api/v1/users", None, {"ADMIN"}),
    ("GET", "/api/v1/users/directory", None, {"ADMIN", "OPERATOR"}),
    ("POST", "/api/v1/users", {"email": "z@example.com", "full_name": "Z", "password": "Sup3r-Secret!"}, {"ADMIN"}),
    ("GET", "/api/v1/security/events", None, {"ADMIN"}),
    ("GET", "/api/v1/security/alerts", None, {"ADMIN"}),
    ("GET", "/api/v1/security/audit", None, {"ADMIN"}),
    ("GET", "/api/v1/policy", None, {"ADMIN", "OPERATOR"}),
    ("GET", "/api/v1/analytics/overview", None, {"ADMIN", "VIEWER"}),
    ("GET", "/api/v1/evaluation/runs", None, {"ADMIN", "VIEWER"}),
    ("GET", "/api/v1/system/health", None, {"ADMIN"}),
    ("GET", "/api/v1/enrolments/requests", None, {"ADMIN", "OPERATOR"}),
    ("GET", "/api/v1/documents", None, {"ADMIN", "OPERATOR"}),
    ("GET", "/api/v1/history/sessions", None, {"ADMIN", "OPERATOR", "VIEWER"}),
]


@pytest.mark.parametrize("method,path,body,allowed", MATRIX)
def test_permission_matrix(client, method, path, body, allowed):
    for role in RoleEnum:
        u = make_user(role)
        h = login(client, u.email)
        r = client.request(method, path, json=body, headers=h)
        if role.value in allowed:
            assert r.status_code < 400, (role, path, r.status_code, r.text)
        else:
            assert r.status_code == 403, (role, path, r.status_code)


def test_permission_denials_are_logged(client):
    v = make_user(RoleEnum.VIEWER)
    client.get("/api/v1/users", headers=login(client, v.email))
    ev = client.get("/api/v1/security/events?event_type=PERMISSION_DENIED", headers=admin_headers(client)).json()
    assert ev["total"] >= 1


def test_viewer_history_has_no_biometric_scores(client):
    h = admin_headers(client)
    t = make_user(RoleEnum.OPERATOR)
    enrol_direct(t.id)
    drive(client, h, start_verify(client, h, t.id), identity=1)
    v = make_user(RoleEnum.VIEWER)
    vh = login(client, v.email)
    rows = client.get("/api/v1/history/sessions", headers=vh).json()["sessions"]
    assert rows and "final_distance" not in rows[0]
    detail = client.get(f"/api/v1/history/sessions/{rows[0]['session_id']}", headers=vh).json()
    assert all("distance" not in e for e in detail["events"])
    assert client.get(f"/api/v1/history/sessions/{rows[0]['session_id']}", headers=h).json()["final_distance"] is not None
