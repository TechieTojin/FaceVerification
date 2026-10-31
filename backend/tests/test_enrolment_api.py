"""Enrolment authorisation, consent, capture, approval workflow, duplicate detection."""
from __future__ import annotations

import pytest

from backend.database import SessionLocal
from backend.models.biometric import BiometricEnrolment, EnrolmentRequest
from backend.models.user import RoleEnum
from backend.services.biometric.providers import fake_identity_vector
from backend.services.biometric.templates import cosine_distance, decrypt_vectors
from backend.tests.helpers import (
    PASSWORD, admin_headers, admin_id, drive, enrol_direct, login, make_user, start_verify,
)


def consent(client, h, user_id, password=PASSWORD):
    return client.post(f"/api/v1/enrolments/{user_id}/consent", json={"subject_password": password, "accept": True},
                       headers=h)


def capture(client, h, user_id, identity=1):
    r = client.post(f"/api/v1/enrolments/{user_id}/sessions", headers=h)
    assert r.status_code == 201, r.text
    final, _ = drive(client, h, r.json(), identity=identity)
    return final


def template_identity_distance(user_id, identity):
    db = SessionLocal()
    try:
        enr = db.query(BiometricEnrolment).filter(BiometricEnrolment.user_id == user_id).one()
        return cosine_distance(decrypt_vectors(enr.embedding_encrypted).vector, fake_identity_vector(identity))
    finally:
        db.close()


@pytest.fixture
def operator(client):
    op = make_user(RoleEnum.OPERATOR)
    return op, login(client, op.email)


def test_full_enrolment_activates_template(client, operator):
    _, h = operator
    subject = make_user(RoleEnum.OPERATOR)
    assert consent(client, h, subject.id).status_code == 200
    final = capture(client, h, subject.id, identity=3)
    assert final["final_outcome"] == "ENROLMENT_CAPTURED"
    assert final["frame"]["activated"] is True and final["frame"]["request_status"] == "AUTO_APPROVED"
    st = client.get(f"/api/v1/enrolments/{subject.id}", headers=h).json()
    assert st["status"] == "ACTIVE" and st["usable"] and st["sample_count"] == 3
    assert template_identity_distance(subject.id, 3) < 1e-4
    # And the new template verifies the same face.
    v, _ = drive(client, h, start_verify(client, h, subject.id), identity=3)
    assert v["final_outcome"] == "VERIFIED"


def test_consent_requires_subject_password(client, operator):
    _, h = operator
    subject = make_user(RoleEnum.OPERATOR)
    r = consent(client, h, subject.id, password="wrong-password")
    assert r.status_code == 403 and r.json()["code"] == "ATTESTATION_FAILED"
    r = client.post(f"/api/v1/enrolments/{subject.id}/sessions", headers=h)
    assert r.status_code == 409 and r.json()["code"] == "CONSENT_REQUIRED"
    r = client.post(f"/api/v1/enrolments/{subject.id}/consent", json={"subject_password": PASSWORD, "accept": False},
                    headers=h)
    assert r.status_code == 422


def test_repeated_attestation_failures_raise_alert(client, operator):
    op, h = operator
    subject = make_user(RoleEnum.OPERATOR)
    for _ in range(4):
        consent(client, h, subject.id, password="nope-nope-nope")
    alerts = client.get("/api/v1/security/alerts", headers=admin_headers(client)).json()["alerts"]
    assert any(a["rule"] == "CONSENT_ATTESTATION_FAILURES" and a["subject_id"] == op.id for a in alerts)


def test_operator_cannot_enrol_or_replace_admin_template(client, operator):
    op, h = operator
    aid = admin_id()
    r = client.post(f"/api/v1/enrolments/{aid}/consent", json={"subject_password": "Admin-Test-Pass-1", "accept": True},
                    headers=h)
    assert r.status_code == 403 and r.json()["code"] == "PRIVILEGED_TARGET"
    r = client.post(f"/api/v1/enrolments/{aid}/sessions", headers=h)
    assert r.status_code == 403
    alerts = client.get("/api/v1/security/alerts", headers=admin_headers(client)).json()["alerts"]
    assert any(a["rule"] == "UNAUTHORIZED_ENROLMENT_ATTEMPT" and a["subject_id"] == op.id for a in alerts)


def test_viewer_cannot_modify_biometrics(client):
    viewer = make_user(RoleEnum.VIEWER)
    h = login(client, viewer.email)
    subject = make_user(RoleEnum.OPERATOR)
    assert consent(client, h, subject.id).status_code == 403
    assert client.post(f"/api/v1/enrolments/{subject.id}/sessions", headers=h).status_code == 403
    assert client.request("DELETE", f"/api/v1/enrolments/{subject.id}", json={}, headers=h).status_code == 403


def test_reenrolment_requires_approval_and_keeps_old_template_until_approved(client, operator):
    op, h = operator
    subject = make_user(RoleEnum.OPERATOR)
    consent(client, h, subject.id)
    capture(client, h, subject.id, identity=3)
    final = capture(client, h, subject.id, identity=4)       # attempted replacement with another face
    assert final["frame"]["activated"] is False and final["frame"]["request_status"] == "PENDING"
    assert "EXISTING_TEMPLATE" in final["frame"]["approval_reasons"]
    assert template_identity_distance(subject.id, 3) < 1e-4   # unchanged
    # A second pending request is blocked.
    r = client.post(f"/api/v1/enrolments/{subject.id}/sessions", headers=h)
    assert r.status_code == 409 and r.json()["code"] == "REQUEST_PENDING"
    # Operator cannot approve.
    req_id = final["frame"]["request_id"]
    assert client.post(f"/api/v1/enrolments/requests/{req_id}/approve", json={}, headers=h).status_code == 403
    ah = admin_headers(client)
    pending = client.get("/api/v1/enrolments/requests?status=PENDING", headers=ah).json()["requests"]
    assert [p["id"] for p in pending] == [req_id]
    r = client.post(f"/api/v1/enrolments/requests/{req_id}/approve", json={"note": "verified in person"}, headers=ah)
    assert r.status_code == 200 and r.json()["status"] == "APPROVED"
    assert template_identity_distance(subject.id, 4) < 1e-4
    alerts = client.get("/api/v1/security/alerts", headers=ah).json()["alerts"]
    assert any(a["rule"] == "TEMPLATE_REPLACEMENT_REQUESTED" for a in alerts)


def test_reject_request_discards_candidate(client, operator):
    _, h = operator
    subject = make_user(RoleEnum.OPERATOR)
    consent(client, h, subject.id)
    capture(client, h, subject.id, identity=3)
    req_id = capture(client, h, subject.id, identity=4)["frame"]["request_id"]
    ah = admin_headers(client)
    assert client.post(f"/api/v1/enrolments/requests/{req_id}/reject", json={"note": "no"}, headers=ah).json()["status"] == "REJECTED"
    db = SessionLocal()
    assert db.query(EnrolmentRequest).filter(EnrolmentRequest.id == req_id).one().candidate_template_encrypted is None
    db.close()
    assert template_identity_distance(subject.id, 3) < 1e-4
    assert client.post(f"/api/v1/enrolments/requests/{req_id}/approve", json={}, headers=ah).status_code == 409


def test_four_eyes_policy(client):
    ah = admin_headers(client)
    assert client.put("/api/v1/policy", json={"changes": {"require_four_eyes": True}}, headers=ah).status_code == 200
    subject = make_user(RoleEnum.OPERATOR)
    consent(client, ah, subject.id)
    capture(client, ah, subject.id, identity=3)
    req_id = capture(client, ah, subject.id, identity=4)["frame"]["request_id"]
    r = client.post(f"/api/v1/enrolments/requests/{req_id}/approve", json={}, headers=ah)
    assert r.status_code == 409 and r.json()["code"] == "FOUR_EYES"
    admin2 = make_user(RoleEnum.ADMIN)
    r = client.post(f"/api/v1/enrolments/requests/{req_id}/approve", json={}, headers=login(client, admin2.email))
    assert r.status_code == 200


def test_duplicate_face_across_users_requires_approval(client, operator):
    _, h = operator
    a = make_user(RoleEnum.OPERATOR)
    b = make_user(RoleEnum.OPERATOR)
    enrol_direct(a.id, identity=5)
    consent(client, h, b.id)
    final = capture(client, h, b.id, identity=5)
    assert final["frame"]["request_status"] == "PENDING"
    assert "DUPLICATE_SUSPECTED" in final["frame"]["approval_reasons"]
    ah = admin_headers(client)
    req = client.get("/api/v1/enrolments/requests?status=PENDING", headers=ah).json()["requests"][0]
    assert req["duplicate_user_ids"] == [a.id]
    op_view = client.get("/api/v1/enrolments/requests", headers=h).json()["requests"][0]
    assert "duplicate_user_ids" not in op_view        # identities of other people not disclosed to operators
    alerts = client.get("/api/v1/security/alerts", headers=ah).json()["alerts"]
    assert any(a_["rule"] == "DUPLICATE_ENROLMENT_SUSPECTED" for a_ in alerts)


def test_self_enrolment_by_operator_needs_approval(client, operator):
    op, h = operator
    consent(client, h, op.id)
    final = capture(client, h, op.id)
    assert final["frame"]["request_status"] == "PENDING" and "SELF_ENROLMENT" in final["frame"]["approval_reasons"]


def test_reenrolment_requested_by_admin_skips_existing_template_rule(client, operator):
    _, h = operator
    subject = make_user(RoleEnum.OPERATOR)
    consent(client, h, subject.id)
    capture(client, h, subject.id, identity=3)
    ah = admin_headers(client)
    assert client.post(f"/api/v1/enrolments/{subject.id}/request-reenrolment", json={"reason": "photo aged"},
                       headers=ah).json()["reenrolment_requested"] is True
    final = capture(client, h, subject.id, identity=4)
    assert final["frame"]["activated"] is True
    assert template_identity_distance(subject.id, 4) < 1e-4


def test_subject_can_withdraw_own_consent(client):
    subject = make_user(RoleEnum.VIEWER)
    enrol_direct(subject.id)
    sh = login(client, subject.email)
    r = client.post(f"/api/v1/enrolments/{subject.id}/consent/withdraw", json={"reason": "no longer"}, headers=sh)
    assert r.status_code == 200 and r.json()["status"] == "WITHDRAWN" and not r.json()["is_enrolled"]
    other = make_user(RoleEnum.OPERATOR)
    enrol_direct(other.id)
    r = client.post(f"/api/v1/enrolments/{other.id}/consent/withdraw", json={}, headers=sh)
    assert r.status_code == 403
    hist = client.get(f"/api/v1/enrolments/{subject.id}", headers=sh).json()["consent_history"]
    assert hist[0]["granted"] is False


def test_revoke_deletes_template_and_is_audited(client):
    ah = admin_headers(client)
    subject = make_user(RoleEnum.OPERATOR)
    enrol_direct(subject.id)
    r = client.request("DELETE", f"/api/v1/enrolments/{subject.id}", json={"reason": "left company"}, headers=ah)
    assert r.json()["status"] == "REVOKED" and not r.json()["is_enrolled"]
    audit = client.get(f"/api/v1/users/{subject.id}/audit", headers=ah).json()["entries"]
    assert audit[0]["action"] == "ENROLMENT_REVOKED"


def test_inconsistent_samples_rejected(client, operator, monkeypatch):
    from backend.services import enrolment_service
    from backend.services.biometric.templates import BuiltTemplate
    import numpy as np
    _, h = operator
    subject = make_user(RoleEnum.OPERATOR)
    consent(client, h, subject.id)
    monkeypatch.setattr(enrolment_service, "build_template",
                        lambda s: BuiltTemplate(vector=np.ones(512, dtype=np.float32) / np.sqrt(512), consistency=0.9,
                                                sample_count=len(s)))
    final = capture(client, h, subject.id)
    assert final["final_outcome"] == "ERROR" and final["frame"]["reason"] == "INCONSISTENT_SAMPLES"
    st = client.get(f"/api/v1/enrolments/{subject.id}", headers=h).json()
    assert st["status"] != "ACTIVE"
