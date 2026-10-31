"""Multi-organisation isolation: one organisation can never read or change another's data."""
from __future__ import annotations

from backend.database import SessionLocal
from backend.models.organization import Membership
from backend.models.user import RoleEnum
from backend.models.verification import SecurityAlert
from backend.services import notification_service
from backend.tests.helpers import (
    PASSWORD, add_member, admin_headers, default_org_id, drive, enrol_direct, login, make_org, make_user,
    start_verify,
)


def _two_orgs(client):
    """Org A = default org (seeded admin). Org B with its own admin, operator and enrolled subject."""
    org_a = default_org_id()
    org_b = make_org("Beta Corp", "beta")
    a_subject = make_user(RoleEnum.VIEWER, name="Alpha Subject")
    enrol_direct(a_subject.id, identity=3)
    b_admin = make_user(RoleEnum.ADMIN, name="Beta Admin", org_id=org_b)
    b_subject = make_user(RoleEnum.VIEWER, name="Beta Subject", org_id=org_b)
    enrol_direct(b_subject.id, identity=4, org_id=org_b)
    return org_a, org_b, a_subject, b_admin, b_subject


def test_users_templates_and_enrolments_are_isolated(client):
    org_a, org_b, a_subject, b_admin, b_subject = _two_orgs(client)
    hb = login(client, b_admin.email)
    ha = admin_headers(client)

    ids_b = {u["id"] for u in client.get("/api/v1/users", headers=hb).json()["users"]}
    assert a_subject.id not in ids_b and b_subject.id in ids_b
    assert client.get(f"/api/v1/users/{a_subject.id}", headers=hb).status_code == 404
    assert client.get("/api/v1/users/directory", headers=hb).json()["users"][0]["id"] != a_subject.id
    assert client.patch(f"/api/v1/users/{a_subject.id}", json={"role": "ADMIN"}, headers=hb).status_code == 404
    assert client.post(f"/api/v1/users/{a_subject.id}/disable", headers=hb).status_code == 404
    assert client.post(f"/api/v1/users/{a_subject.id}/reset-password", json={"new_password": "Xx-123456789"},
                       headers=hb).status_code == 404
    assert client.delete(f"/api/v1/users/{a_subject.id}", headers=hb).status_code == 404

    # Biometric records of org A are invisible and immutable from org B.
    assert client.get(f"/api/v1/enrolments/{a_subject.id}", headers=hb).status_code == 404
    assert client.request("DELETE", f"/api/v1/enrolments/{a_subject.id}", json={"reason": "x"},
                          headers=hb).status_code == 404
    assert client.post(f"/api/v1/enrolments/{a_subject.id}/consent/withdraw", json={"reason": "x"},
                       headers=hb).status_code == 404
    assert client.post(f"/api/v1/enrolments/{a_subject.id}/sessions", headers=hb).status_code == 404
    # ...and a verification session in org B cannot target an org A subject.
    r = client.post("/api/v1/sessions", json={"target_user_id": a_subject.id}, headers=hb)
    assert r.status_code == 404 and r.json()["code"] == "TARGET_NOT_FOUND"
    # Org A still sees its own subject as enrolled.
    assert client.get(f"/api/v1/enrolments/{a_subject.id}", headers=ha).json()["status"] == "ACTIVE"


def test_sessions_history_export_and_alerts_are_isolated(client):
    org_a, org_b, a_subject, b_admin, b_subject = _two_orgs(client)
    ha = admin_headers(client)
    sess_a = start_verify(client, ha, a_subject.id)
    final_a, _ = drive(client, ha, sess_a, identity=3)
    assert final_a["final_outcome"] == "VERIFIED"

    hb = login(client, b_admin.email)
    sess_b = start_verify(client, hb, b_subject.id)
    final_b, _ = drive(client, hb, sess_b, identity=4)
    assert final_b["final_outcome"] == "VERIFIED"

    # Session object, frames, history detail and CSV export never cross the boundary.
    assert client.get(f"/api/v1/sessions/{sess_a['session_id']}", headers=hb).status_code == 404
    assert client.post(f"/api/v1/sessions/{sess_a['session_id']}/cancel", headers=hb).status_code == 404
    assert client.get(f"/api/v1/history/sessions/{sess_a['session_id']}", headers=hb).status_code == 404
    ids_b = {s["session_id"] for s in client.get("/api/v1/history/sessions", headers=hb).json()["sessions"]}
    assert ids_b == {sess_b["session_id"]}
    csv_b = client.get("/api/v1/history/export.csv", headers=hb).text
    assert sess_a["session_id"] not in csv_b and sess_b["session_id"] in csv_b
    # Cross-tenant session access raised an alert in the *requesting* organisation only.
    alerts_b = client.get("/api/v1/security/alerts", headers=hb).json()["alerts"]
    assert any(a["rule"] == "CROSS_TENANT_ACCESS" for a in alerts_b)
    alerts_a = client.get("/api/v1/security/alerts", headers=ha).json()["alerts"]
    assert not any(a["rule"] == "CROSS_TENANT_ACCESS" for a in alerts_a)
    # Dashboard counts are per organisation.
    assert client.get("/api/v1/dashboard/summary", headers=hb).json()["verified"] == 1
    assert client.get("/api/v1/analytics/overview", headers=hb).json()["totals"]["sessions"] == 1

    # An org-A alert cannot be acknowledged from org B.
    db = SessionLocal()
    alert_a = db.query(SecurityAlert).filter(SecurityAlert.organization_id == org_a).first()
    if alert_a is None:
        alert_a = SecurityAlert(organization_id=org_a, rule="X", title="x", severity="LOW", status="OPEN")
        db.add(alert_a)
        db.commit()
    alert_id = alert_a.id
    db.close()
    assert client.post(f"/api/v1/security/alerts/{alert_id}/acknowledge", json={"note": ""},
                       headers=hb).status_code == 404


def test_policy_audit_documents_and_notifications_are_isolated(client):
    org_a, org_b, a_subject, b_admin, b_subject = _two_orgs(client)
    ha = admin_headers(client)
    hb = login(client, b_admin.email)

    r = client.put("/api/v1/policy", json={"changes": {"challenge_steps": 3}}, headers=hb)
    assert r.status_code == 200 and r.json()["version"] == 2
    assert client.get("/api/v1/policy", headers=ha).json()["values"]["challenge_steps"] == 2   # org A untouched
    assert client.get("/api/v1/policy", headers=hb).json()["values"]["challenge_steps"] == 3
    assert client.get("/api/v1/policy/versions/2", headers=ha).status_code == 404

    audit_b = client.get("/api/v1/security/audit", headers=hb).json()["entries"]
    assert all(e["action"] != "USER_CREATED" or e["target_id"] != a_subject.id for e in audit_b)
    assert any(e["action"] == "POLICY_UPDATED" for e in audit_b)
    assert not any(e["action"] == "POLICY_UPDATED" for e in client.get("/api/v1/security/audit", headers=ha).json()["entries"])

    doc = client.post("/api/v1/documents", json={"consent_given": True, "subject_label": "A doc"}, headers=ha).json()
    assert client.get(f"/api/v1/documents/{doc['id']}", headers=hb).status_code == 404
    assert client.get("/api/v1/documents", headers=hb).json()["total"] == 0
    # Cannot open a document check for a subject of another organisation either.
    r = client.post("/api/v1/documents", json={"consent_given": True, "target_user_id": a_subject.id}, headers=hb)
    assert r.status_code == 404

    db = SessionLocal()
    n = notification_service.notify(db, org_a, a_subject.id, "ACCOUNT_SECURITY", "Org A notice")
    db.commit()
    nid = n.id
    db.close()
    assert client.post(f"/api/v1/notifications/{nid}/read", headers=hb).status_code == 404


def test_organisation_switching_requires_active_membership(client):
    org_a = default_org_id()
    org_b = make_org("Gamma", "gamma")
    org_c = make_org("Delta", "delta")
    user = make_user(RoleEnum.OPERATOR, name="Two Orgs")
    add_member(user.id, org_b, RoleEnum.ADMIN)
    h = login(client, user.email)
    me = client.get("/api/v1/auth/me", headers=h).json()
    assert {o["id"] for o in me["organizations"]} == {org_a, org_b}
    assert me["organization"]["id"] == org_a and me["role"] == "OPERATOR"

    # Not a member of C → refused (and nothing changes).
    assert client.post("/api/v1/auth/switch-org", json={"organization_id": org_c}, headers=h).status_code == 404
    r = client.post("/api/v1/auth/switch-org", json={"organization_id": org_b}, headers=h)
    assert r.status_code == 200
    client.cookies.clear()     # the switch also set the browser cookie; this test uses bearer tokens
    hb = {"Authorization": f"Bearer {r.json()['access_token']}"}
    me_b = client.get("/api/v1/auth/me", headers=hb).json()
    assert me_b["organization"]["id"] == org_b and me_b["role"] == "ADMIN" and "users:manage" in me_b["permissions"]
    assert client.get("/api/v1/users", headers=h).status_code == 403          # still OPERATOR in A
    assert client.get("/api/v1/users", headers=hb).status_code == 200         # ADMIN in B

    # Membership disabled → the org-B token stops working immediately.
    db = SessionLocal()
    db.query(Membership).filter(Membership.user_id == user.id, Membership.organization_id == org_b).update(
        {"status": "DISABLED"})
    db.commit()
    db.close()
    assert client.get("/api/v1/auth/me", headers=hb).status_code == 401
    assert client.get("/api/v1/auth/me", headers=h).status_code == 200


def test_suspended_organisation_loses_access(client):
    org_b = make_org("Epsilon", "epsilon")
    b_admin = make_user(RoleEnum.ADMIN, org_id=org_b)
    hb = login(client, b_admin.email)
    ha = admin_headers(client)
    # Step-up is required for suspension; a password-only session is refused.
    r = client.post(f"/api/v1/organizations/{org_b}/suspend", headers=ha)
    assert r.status_code == 403 and r.json()["code"] == "STEP_UP_REQUIRED"
    from backend.models.organization import Organization
    db = SessionLocal()
    db.query(Organization).filter(Organization.id == org_b).update({"status": "SUSPENDED"})
    db.commit()
    db.close()
    assert client.get("/api/v1/auth/me", headers=hb).status_code == 401
    r = client.post("/api/v1/auth/login", json={"email": b_admin.email, "password": PASSWORD})
    assert r.status_code == 403   # no active organisation left


def test_shared_account_cannot_be_managed_by_one_organisation(client):
    org_b = make_org("Zeta", "zeta")
    b_admin = make_user(RoleEnum.ADMIN, org_id=org_b)
    shared = make_user(RoleEnum.OPERATOR, name="Shared")          # member of default org
    add_member(shared.id, org_b, RoleEnum.OPERATOR)
    hb = login(client, b_admin.email)
    r = client.post(f"/api/v1/users/{shared.id}/reset-password", json={"new_password": "Brand-New-Pass-9"}, headers=hb)
    assert r.status_code == 409
    r = client.patch(f"/api/v1/users/{shared.id}", json={"full_name": "Renamed"}, headers=hb)
    assert r.status_code == 409
    # Disabling only affects the membership in B; the account still signs in to the default org.
    assert client.post(f"/api/v1/users/{shared.id}/disable", headers=hb).status_code == 200
    h = login(client, shared.email)
    assert client.get("/api/v1/auth/me", headers=h).json()["organization"]["id"] == default_org_id()
    assert client.get("/api/v1/users/me-not-found", headers=hb).status_code == 404


def test_invitation_flow_new_and_existing_accounts(client):
    ha = admin_headers(client)
    r = client.post("/api/v1/organizations/current/invitations", json={"email": "new.person@test.local",
                                                                       "role": "VIEWER"}, headers=ha)
    assert r.status_code == 201
    body = r.json()
    assert body["email_status"] == "UNAVAILABLE"          # no SMTP configured: reported, not faked
    token = body["link"].rsplit("/", 1)[1]
    info = client.get(f"/api/v1/invitations/{token}").json()
    assert info["role"] == "VIEWER" and info["account_exists"] is False and "new.person" not in info["email"]
    r = client.post(f"/api/v1/invitations/{token}/accept", json={"full_name": "New Person", "password": "weak"})
    assert r.status_code == 422
    r = client.post(f"/api/v1/invitations/{token}/accept", json={"full_name": "New Person",
                                                                  "password": "Good-Pass-12345"})
    assert r.status_code == 200 and r.json()["session"]["user"]["role"] == "VIEWER"
    client.cookies.clear()
    assert client.post(f"/api/v1/invitations/{token}/accept", json={}).status_code == 410   # one-time

    # Existing account in another organisation must sign in first, then accept.
    org_b = make_org("Eta", "eta")
    b_admin = make_user(RoleEnum.ADMIN, org_id=org_b)
    hb = login(client, b_admin.email)
    existing = make_user(RoleEnum.OPERATOR, name="Existing")
    token2 = client.post("/api/v1/organizations/current/invitations", json={"email": existing.email,
                                                                            "role": "OPERATOR"},
                         headers=hb).json()["link"].rsplit("/", 1)[1]
    assert client.post(f"/api/v1/invitations/{token2}/accept", json={}).json()["code"] == "SIGN_IN_REQUIRED"
    he = login(client, existing.email)
    assert client.post(f"/api/v1/invitations/{token2}/accept", json={}, headers=he).status_code == 200
    me = client.get("/api/v1/auth/me", headers=he).json()
    assert org_b in {o["id"] for o in me["organizations"]}


def test_only_platform_admins_create_organisations(client):
    ha = admin_headers(client)
    op = make_user(RoleEnum.ADMIN, name="Org admin only")      # ADMIN of default org, not platform admin
    ho = login(client, op.email)
    assert client.post("/api/v1/organizations", json={"name": "Nope Inc"}, headers=ho).status_code == 403
    r = client.post("/api/v1/organizations", json={"name": "Platform Made"}, headers=ha)
    assert r.status_code == 403 and r.json()["code"] == "STEP_UP_REQUIRED"   # strong re-auth needed
    listed = client.get("/api/v1/organizations", headers=ho).json()
    assert listed["can_create"] is False and len(listed["organizations"]) == 1
