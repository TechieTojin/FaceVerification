"""History, alerts, documents, evaluation, analytics, health, retention and event bus."""
from __future__ import annotations

import asyncio
import csv
import io
import json
import zipfile
from datetime import datetime, timedelta

import pytest

from backend.database import SessionLocal
from backend.models.documents import DocumentCheck
from backend.models.user import RoleEnum
from backend.models.verification import VerificationEvent, VerificationSession
from backend.services import evaluation_service
from backend.services.inference import runtime
from backend.tests.helpers import (
    admin_headers, drive, enrol_direct, frame, frame_for_step, login, make_user, start_verify, submit,
)


def _verified_and_rejected(client):
    op = make_user(RoleEnum.OPERATOR, name="Op One")
    t = make_user(RoleEnum.OPERATOR, name="=HYPERLINK(evil)")
    enrol_direct(t.id)
    h = login(client, op.email)
    drive(client, h, start_verify(client, h, t.id), identity=1)
    drive(client, h, start_verify(client, h, t.id), identity=2)
    return op, h, t


# ── history ─────────────────────────────────────────────────────────────

def test_history_scoping_filters_and_detail(client):
    op, h, t = _verified_and_rejected(client)
    other = make_user(RoleEnum.OPERATOR)
    assert client.get("/api/v1/history/sessions", headers=login(client, other.email)).json()["total"] == 0
    mine = client.get("/api/v1/history/sessions", headers=h).json()
    assert mine["total"] == 2
    v = client.get("/api/v1/history/sessions?outcome=VERIFIED", headers=h).json()
    assert v["total"] == 1 and v["sessions"][0]["liveness_outcome"] == "PASSED"
    s = client.get("/api/v1/history/sessions?sort=distance", headers=h).json()["sessions"]
    assert s[0]["final_distance"] <= s[1]["final_distance"]
    by_name = client.get("/api/v1/history/sessions?search=Op%20One", headers=admin_headers(client)).json()
    assert by_name["total"] == 2
    d = client.get(f"/api/v1/history/sessions/{v['sessions'][0]['session_id']}", headers=h).json()
    assert d["challenges"] and d["events"] and d["policy_snapshot"]["match_threshold"] == 0.68
    assert client.get(f"/api/v1/history/sessions/{v['sessions'][0]['session_id']}",
                      headers=login(client, other.email)).status_code == 404


def test_history_csv_export_is_safe(client):
    _, h, _ = _verified_and_rejected(client)
    r = client.get("/api/v1/history/export.csv", headers=h)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert len(rows) == 2 and {row["outcome"] for row in rows} == {"VERIFIED", "NOT_MATCHED"}
    assert all(row["target_name"].startswith("'=") for row in rows)   # formula injection neutralised


# ── alerts ──────────────────────────────────────────────────────────────

def test_repeated_failures_raise_alert_then_ack_and_resolve(client):
    ah = admin_headers(client)
    assert client.put("/api/v1/policy", json={"changes": {"alert_failed_verifications_threshold": 2}}, headers=ah).status_code == 200
    op, h, t = _verified_and_rejected(client)
    drive(client, h, start_verify(client, h, t.id), identity=2)      # second failure for the same identity
    alerts = client.get("/api/v1/security/alerts", headers=ah).json()["alerts"]
    a = next(x for x in alerts if x["rule"] == "REPEATED_VERIFICATION_FAILURES")
    assert a["severity"] == "HIGH" and a["status"] == "OPEN"
    assert client.post(f"/api/v1/security/alerts/{a['id']}/resolve", json={"note": ""}, headers=ah).status_code == 422
    assert client.post(f"/api/v1/security/alerts/{a['id']}/acknowledge", json={}, headers=ah).json()["status"] == "ACKNOWLEDGED"
    r = client.post(f"/api/v1/security/alerts/{a['id']}/resolve", json={"note": "Confirmed test traffic"}, headers=ah).json()
    assert r["status"] == "RESOLVED" and r["resolution_note"] == "Confirmed test traffic"
    assert client.post(f"/api/v1/security/alerts/{a['id']}/acknowledge", json={}, headers=ah).status_code == 409
    assert client.post(f"/api/v1/security/alerts/{a['id']}/acknowledge", json={}, headers=h).status_code == 403


# ── documents ───────────────────────────────────────────────────────────

def _doc_flow(client, h, live_identity, doc_identity=1):
    r = client.post("/api/v1/documents", json={"subject_label": "Visitor A", "document_type": "PASSPORT",
                                               "consent_given": True}, headers=h)
    assert r.status_code == 201
    cid = r.json()["id"]
    r = client.post(f"/api/v1/documents/{cid}/image", headers=h,
                    files={"file": ("doc.jpg", frame(identity=doc_identity, size=(1000, 700), width_code=4), "image/jpeg")})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "AWAITING_LIVE_CAPTURE" and r.json()["portrait_template_stored"]
    s = client.post(f"/api/v1/documents/{cid}/session", headers=h)
    assert s.status_code == 201
    drive(client, h, s.json(), identity=live_identity)
    return client.get(f"/api/v1/documents/{cid}", headers=h).json()


def test_document_match_never_claims_authenticity(client):
    op = make_user(RoleEnum.OPERATOR)
    h = login(client, op.email)
    d = _doc_flow(client, h, live_identity=1)
    assert d["face_match_result"] == "MATCH" and d["liveness_passed"] is True
    assert d["authenticity_status"] == "NOT_PERFORMED"
    assert d["overall_decision"] == "FACE_MATCH_ONLY"
    assert d["portrait_template_stored"] is False and d["purged_at"]   # minimised after use
    d2 = _doc_flow(client, h, live_identity=2)
    assert d2["face_match_result"] == "NO_MATCH" and d2["overall_decision"] == "NOT_VERIFIED"


def test_document_requires_consent_and_portrait(client):
    op = make_user(RoleEnum.OPERATOR)
    h = login(client, op.email)
    assert client.post("/api/v1/documents", json={"consent_given": False}, headers=h).status_code == 400
    cid = client.post("/api/v1/documents", json={"consent_given": True}, headers=h).json()["id"]
    r = client.post(f"/api/v1/documents/{cid}/image", headers=h,
                    files={"file": ("doc.jpg", frame(face_count=0, size=(1000, 700)), "image/jpeg")})
    assert r.status_code == 422 and r.json()["code"] == "NO_PORTRAIT"
    assert client.post(f"/api/v1/documents/{cid}/session", headers=h).status_code == 409
    other = make_user(RoleEnum.OPERATOR)
    assert client.get(f"/api/v1/documents/{cid}", headers=login(client, other.email)).status_code == 404
    assert client.delete(f"/api/v1/documents/{cid}", headers=h).json()["status"] == "DELETED"


def test_document_portrait_purged_on_retention(client):
    from backend.services.maintenance_service import run_maintenance
    op = make_user(RoleEnum.OPERATOR)
    h = login(client, op.email)
    cid = client.post("/api/v1/documents", json={"consent_given": True}, headers=h).json()["id"]
    client.post(f"/api/v1/documents/{cid}/image", headers=h,
                files={"file": ("doc.jpg", frame(size=(1000, 700), width_code=4), "image/jpeg")})
    db = SessionLocal()
    db.query(DocumentCheck).filter(DocumentCheck.id == cid).update(
        {"retention_expires_at": datetime.utcnow() - timedelta(minutes=1)})
    db.commit()
    db.close()
    assert run_maintenance()["documents_purged"] == 1
    d = client.get(f"/api/v1/documents/{cid}", headers=h).json()
    assert d["status"] == "EXPIRED" and not d["portrait_template_stored"]


# ── evaluation ──────────────────────────────────────────────────────────

def test_metric_maths_on_known_distributions():
    genuine = [0.1, 0.2, 0.3, 0.4]
    impostor = [0.5, 0.6, 0.7, 0.8]
    m = evaluation_service.compute_metrics(genuine, impostor, 0.45)
    assert m["at_policy_threshold"] == {"threshold": 0.45, "far": 0.0, "frr": 0.0}
    assert m["eer"] == pytest.approx(0.0, abs=1e-9)
    overlap = evaluation_service.compute_metrics([0.2, 0.4, 0.6, 0.8], [0.3, 0.5, 0.7, 0.9], 0.55)
    assert overlap["at_policy_threshold"]["far"] == 0.5 and overlap["at_policy_threshold"]["frr"] == 0.5
    assert 0.2 < overlap["eer"] < 0.6
    assert any("unreliable" in w for w in m["warnings"])


def _zip(groups):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for ident, images in groups.items():
            for i, data in enumerate(images):
                z.writestr(f"{ident}/{i}.jpg", data)
    return buf.getvalue()


def test_dataset_validation():
    with pytest.raises(evaluation_service.EvaluationInputError):
        evaluation_service.parse_dataset(b"not a zip")
    with pytest.raises(evaluation_service.EvaluationInputError):
        evaluation_service.parse_dataset(_zip({"only": [frame(), frame()]}))


def test_evaluation_run_end_to_end(client):
    import time
    ah = admin_headers(client)
    data = _zip({"alice": [frame(identity=1), frame(identity=13)], "bob": [frame(identity=2), frame(identity=2)],
                 "carl": [frame(identity=3), frame(face_count=0)]})
    r = client.post("/api/v1/evaluation/runs", headers=ah, data={"label": "synthetic"},
                    files={"dataset": ("set.zip", data, "application/zip")})
    assert r.status_code == 202
    rid = r.json()["id"]
    for _ in range(50):
        run = client.get(f"/api/v1/evaluation/runs/{rid}", headers=ah).json()
        if run["status"] != "RUNNING":
            break
        time.sleep(0.1)
    assert run["status"] == "COMPLETED", run
    assert run["skipped_images"] == 1 and run["genuine_pairs"] == 2 and run["impostor_pairs"] == 8
    assert run["results"]["at_policy_threshold"]["far"] == 0.0
    viewer = make_user(RoleEnum.VIEWER)
    assert client.post("/api/v1/evaluation/runs", headers=login(client, viewer.email),
                       files={"dataset": ("set.zip", data, "application/zip")}).status_code == 403


# ── analytics / dashboard ───────────────────────────────────────────────

def test_dashboard_and_analytics_reflect_real_data(client):
    _verified_and_rejected(client)
    ah = admin_headers(client)
    d = client.get("/api/v1/dashboard/summary", headers=ah).json()
    assert d["verified"] == 1 and d["not_matched"] == 1 and d["completed_verifications"] == 2
    assert d["health"]["database"]["ok"] is True and d["health"]["models"]["state"] == "READY"
    assert d["latency_last_hour_ms"]["samples"] > 0 and d["recent_sessions"]
    a = client.get("/api/v1/analytics/overview?days=7", headers=ah).json()
    assert a["session_completion_rate"] == 1.0 and a["verified_rate"] == 0.5
    assert a["latency_ms"]["p95"] is not None and len(a["daily"]) == 8


def test_dashboard_health_is_not_hardcoded(client, monkeypatch):
    ah = admin_headers(client)
    monkeypatch.setattr(runtime, "warmup_state", "FAILED")
    d = client.get("/api/v1/dashboard/summary", headers=ah).json()
    assert d["health"]["models"]["ok"] is False and d["health"]["ready"] is False
    assert client.get("/api/v1/health/ready").status_code == 503


# ── health / retention / bus ────────────────────────────────────────────

def test_health_endpoints(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"
    r = client.get("/api/v1/health/ready")
    assert r.status_code == 200 and r.json()["checks"]["migrations"]["up_to_date"] is True
    sysh = client.get("/api/v1/system/health", headers=admin_headers(client)).json()
    assert sysh["inference"]["provider"] == "fake" and sysh["database_backend"] == "sqlite"


def test_retention_purges_old_events(client):
    from backend.services.maintenance_service import run_maintenance
    _verified_and_rejected(client)
    db = SessionLocal()
    old = datetime.utcnow() - timedelta(days=400)
    db.query(VerificationEvent).update({"timestamp": old})
    db.query(VerificationSession).update({"completed_at": old})
    db.commit()
    db.close()
    out = run_maintenance()
    assert out["events_deleted"] > 0 and out["sessions_deleted"] == 2


def test_event_bus_delivers_to_matching_channels():
    from backend.core.events import EventBus

    async def scenario():
        bus = EventBus()
        sub = bus.subscribe({"alerts", "notifications"}, "org-a", "user-1")
        other_org = bus.subscribe({"alerts"}, "org-b", "user-2")
        bus.publish("alerts", "alert", {"x": 1}, org_id="org-a")
        bus.publish("sessions", "session_created", {"x": 2}, org_id="org-a")    # channel not subscribed
        bus.publish("alerts", "alert", {"x": 3}, org_id="org-b")                # other organisation
        bus.publish("alerts", "alert", {"x": 4})                                # no organisation → never delivered
        bus.publish("notifications", "notification", {"x": 5}, org_id="org-a")  # unaddressed → dropped
        bus.publish("notifications", "notification", {"x": 6}, org_id="org-a", user_id="user-9")  # someone else
        bus.publish("notifications", "notification", {"x": 7}, org_id="org-a", user_id="user-1")
        got = [json.loads(await asyncio.wait_for(sub.queue.get(), 1))["data"]["x"] for _ in range(2)]
        assert got == [1, 7]
        assert json.loads(await asyncio.wait_for(other_org.queue.get(), 1))["data"] == {"x": 3}
        await asyncio.sleep(0.05)
        assert sub.queue.empty() and other_org.queue.empty()
        bus.unsubscribe(sub)
        bus.unsubscribe(other_org)
        assert bus.subscriber_count == 0
    asyncio.run(scenario())


def test_security_headers(client):
    r = client.get("/api/v1/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Cache-Control"] == "no-store" and r.headers.get("X-Request-ID")
