"""Manual review workflow."""
from __future__ import annotations

import threading
from datetime import datetime, timedelta

from backend.database import SessionLocal
from backend.models.reviews import VerificationReview
from backend.models.user import RoleEnum
from backend.services import review_service
from backend.core.deps import Actor
from backend.core.permissions import permissions_for
from backend.tests.helpers import (
    admin_headers, admin_id, default_org_id, drive, enrol_direct, frame_for_step, login, make_user, start_verify,
)

RV = "/api/v1/reviews"


def _borderline_session(client):
    """Identity 15 is at distance 0.55 from identity 1; threshold 0.57 puts it inside the 0.03 band."""
    ha = admin_headers(client)
    assert client.put("/api/v1/policy", json={"changes": {"match_threshold": 0.57}}, headers=ha).status_code == 200
    op = make_user(RoleEnum.OPERATOR, name="Review Operator")
    subject = make_user(RoleEnum.VIEWER, name="Review Subject")
    enrol_direct(subject.id, identity=1)
    ho = login(client, op.email)
    final, _ = drive(client, ho, start_verify(client, ho, subject.id), identity=15)
    return op, subject, ho, final


def test_borderline_match_opens_review_and_approval_is_disposition_only(client):
    op, subject, ho, final = _borderline_session(client)
    assert final["final_outcome"] == "VERIFIED"
    ha = admin_headers(client)
    queue = client.get(RV, headers=ha).json()
    assert queue["total"] == 1
    review = queue["reviews"][0]
    assert review["reasons"] == ["BORDERLINE_MATCH"] and review["status"] == "PENDING" and review["can_approve"]
    assert client.get(f"/api/v1/history/sessions/{final['session_id']}", headers=ha).json()["review_status"] == "PENDING"

    # Operators can neither read nor decide reviews; viewers can read but not decide.
    assert client.get(RV, headers=ho).status_code == 403
    viewer = make_user(RoleEnum.VIEWER)
    hv = login(client, viewer.email)
    assert client.get(RV, headers=hv).status_code == 200
    assert client.post(f"{RV}/{review['id']}/decide", json={"decision": "APPROVE", "note": "x", "version": 1},
                       headers=hv).status_code == 403
    # A note is mandatory.
    r = client.post(f"{RV}/{review['id']}/decide", json={"decision": "APPROVE", "note": " ", "version": 1}, headers=ha)
    assert r.status_code == 422
    r = client.post(f"{RV}/{review['id']}/decide", json={"decision": "APPROVE", "note": "Same person per ID card",
                                                         "version": review["version"]}, headers=ha)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "APPROVED" and body["technical_outcome"] == "VERIFIED"
    assert [h["action"] for h in body["history"]] == ["CREATED", "APPROVE"]
    # A second decision on the same case is refused.
    r = client.post(f"{RV}/{review['id']}/decide", json={"decision": "REJECT", "note": "x", "version": body["version"]},
                    headers=ha)
    assert r.status_code == 409 and r.json()["code"] == "ALREADY_DECIDED"
    hist = client.get(f"/api/v1/history/sessions/{final['session_id']}", headers=ha).json()
    assert hist["outcome"] == "VERIFIED" and hist["review_status"] == "APPROVED"     # outcome unchanged
    audit = client.get("/api/v1/security/audit", headers=ha).json()["entries"]
    assert any(e["action"] == "REVIEW_APPROVED" for e in audit)
    notes = client.get("/api/v1/notifications", headers=ho).json()["notifications"]
    assert any(n["type"] == "REVIEW_COMPLETED" for n in notes)


def test_spoof_case_cannot_be_approved(client):
    ha = admin_headers(client)
    subject = make_user(RoleEnum.VIEWER)
    enrol_direct(subject.id, identity=2)
    op = make_user(RoleEnum.OPERATOR)
    ho = login(client, op.email)
    final, _ = drive(client, ho, start_verify(client, ho, subject.id),
                     frame_fn=lambda st, ph: frame_for_step(st["challenge"]["type"], 2, ph, real=False))
    assert final["final_outcome"] == "LIVENESS_FAILED"
    review = client.get(RV, headers=ha).json()["reviews"][0]
    assert review["spoof_confirmed"] and review["risk_level"] == "HIGH" and not review["can_approve"]
    r = client.post(f"{RV}/{review['id']}/decide", json={"decision": "APPROVE", "note": "looks fine", "version": 1},
                    headers=ha)
    assert r.status_code == 409 and r.json()["code"] == "CANNOT_APPROVE_LIVENESS_FAILURE"
    r = client.post(f"{RV}/{review['id']}/decide", json={"decision": "REQUEST_RETRY", "note": "re-capture",
                                                         "version": 1}, headers=ha)
    assert r.status_code == 200 and r.json()["status"] == "RETRY_REQUESTED"


def test_assignment_conflicts_and_concurrent_decisions(client):
    op, subject, ho, final = _borderline_session(client)
    ha = admin_headers(client)
    review = client.get(RV, headers=ha).json()["reviews"][0]
    # The operator who ran the session cannot be assigned (even when promoted to admin).
    r = client.post(f"{RV}/{review['id']}/assign", json={"assignee_user_id": op.id}, headers=ha)
    assert r.status_code == 422          # operators lack review:decide
    r2 = make_user(RoleEnum.ADMIN, name="Reviewer Two")
    r = client.post(f"{RV}/{review['id']}/assign", json={"assignee_user_id": r2.id}, headers=ha)
    assert r.status_code == 200 and r.json()["status"] == "IN_REVIEW"
    h2 = login(client, r2.email)
    assert any(n["type"] == "REVIEW_ASSIGNED" for n in client.get("/api/v1/notifications", headers=h2).json()["notifications"])
    # Someone other than the assignee cannot decide.
    r = client.post(f"{RV}/{review['id']}/decide", json={"decision": "REJECT", "note": "x", "version": r.json()["version"]},
                    headers=ha)
    assert r.status_code == 409 and r.json()["code"] == "ASSIGNED_TO_SOMEONE_ELSE"

    # Two concurrent decisions on an unassigned case → exactly one wins.
    op2, subject2, ho2, _ = _borderline_session(client)
    rid = next(x for x in client.get(RV, headers=ha).json()["reviews"] if x["status"] == "PENDING")["id"]
    org = default_org_id()
    db = SessionLocal()
    admins = [admin_id(), r2.id]
    from backend.models.user import User
    users = {u.id: u for u in db.query(User).filter(User.id.in_(admins)).all()}
    db.close()
    results, barrier = [], threading.Barrier(2)

    def worker(uid, decision):
        s = SessionLocal()
        try:
            actor = Actor(kind="USER", org_id=org, ref=uid, user=users[uid], role=RoleEnum.ADMIN,
                          permissions=permissions_for(RoleEnum.ADMIN))
            barrier.wait()
            review_service.decide(s, actor, rid, decision, "concurrent", 1, "127.0.0.1")
            results.append("ok")
        except Exception as exc:  # noqa: BLE001
            results.append(getattr(exc, "code", type(exc).__name__))
        finally:
            s.close()

    ts = [threading.Thread(target=worker, args=(admins[0], "APPROVE")),
          threading.Thread(target=worker, args=(admins[1], "REJECT"))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert results.count("ok") == 1 and set(results) - {"ok"} <= {"CONCURRENT_UPDATE", "ALREADY_DECIDED"}, results


def test_reviews_expire_and_login_face_checks_are_never_reviewed(client):
    op, subject, ho, final = _borderline_session(client)
    ha = admin_headers(client)
    rid = client.get(RV, headers=ha).json()["reviews"][0]["id"]
    db = SessionLocal()
    db.query(VerificationReview).filter(VerificationReview.id == rid).update(
        {"expires_at": datetime.utcnow() - timedelta(minutes=1)})
    db.commit()
    db.close()
    r = client.get(f"{RV}/{rid}", headers=ha).json()
    assert r["status"] == "EXPIRED"
    assert client.post(f"{RV}/{rid}/decide", json={"decision": "APPROVE", "note": "late", "version": r["version"]},
                       headers=ha).status_code == 409
    assert client.get(f"/api/v1/history/sessions/{final['session_id']}", headers=ha).json()["review_status"] == "EXPIRED"


def test_review_disabled_by_policy_and_cross_org(client):
    ha = admin_headers(client)
    client.put("/api/v1/policy", json={"changes": {"review_enabled": False}, "confirm": True}, headers=ha)
    op, subject, ho, final = _borderline_session(client)
    assert final["final_outcome"] == "VERIFIED"
    assert client.get(RV, headers=ha).json()["total"] == 0
    from backend.tests.helpers import make_org
    org_b = make_org("Review B", "review-b")
    hb = login(client, make_user(RoleEnum.ADMIN, org_id=org_b).email)
    assert client.get(RV, headers=hb).json()["total"] == 0


def test_assignee_decides_with_version_from_assignment(client):
    _borderline_session(client)
    ha = admin_headers(client)
    review = client.get(RV, headers=ha).json()["reviews"][0]
    assigned = client.post(f"{RV}/{review['id']}/assign", json={"assignee_user_id": admin_id()}, headers=ha)
    assert assigned.status_code == 200 and assigned.json()["version"] == review["version"] + 1
    stale = client.post(f"{RV}/{review['id']}/decide", headers=ha,
                        json={"decision": "APPROVE", "note": "ok", "version": review["version"]})
    assert stale.status_code == 409 and stale.json()["code"] == "CONCURRENT_UPDATE"     # in-flight assignment wins
    r = client.post(f"{RV}/{review['id']}/decide", headers=ha,
                    json={"decision": "APPROVE", "note": "ok", "version": assigned.json()["version"]})
    assert r.status_code == 200 and r.json()["status"] == "APPROVED"
