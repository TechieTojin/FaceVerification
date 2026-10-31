"""Capture-session API: decisions, expiry, ownership, replay, concurrency, failure handling."""
from __future__ import annotations

import threading
from datetime import datetime, timedelta

import pytest
from sqlalchemy.exc import OperationalError

from backend.database import SessionLocal
from backend.models.biometric import BiometricEnrolment
from backend.models.user import RoleEnum
from backend.models.verification import LivenessChallenge, VerificationEvent, VerificationSession
from backend.services import capture_service
from backend.services.inference import InferenceBusy, runtime
from backend.tests.helpers import (
    admin_headers, drive, enrol_direct, frame, frame_for_step, login, make_user, start_verify, submit,
)


@pytest.fixture
def setup(client):
    op = make_user(RoleEnum.OPERATOR)
    target = make_user(RoleEnum.OPERATOR)
    enrol_direct(target.id, identity=1)
    return login(client, op.email), target, op


def _session_row(sid):
    db = SessionLocal()
    try:
        return db.query(VerificationSession).filter(VerificationSession.id == sid).one()
    finally:
        db.close()


def test_correct_face_verifies_with_server_side_challenge(client, setup):
    h, target, _ = setup
    final, hist = drive(client, h, start_verify(client, h, target.id), identity=1)
    assert final["final_outcome"] == "VERIFIED" and final["status"] == "CLOSED"
    assert final["final_distance"] == pytest.approx(0.0, abs=1e-4) and final["threshold"] == 0.68
    assert "frame_token" not in final
    db = SessionLocal()
    challenges = db.query(LivenessChallenge).filter(LivenessChallenge.session_id == final["session_id"]).all()
    assert len(challenges) == 4 and all(c.status == "PASSED" for c in challenges)
    events = db.query(VerificationEvent).filter(VerificationEvent.session_id == final["session_id"]).count()
    db.close()
    assert events == len(hist)


def test_incorrect_face_not_matched(client, setup):
    h, target, _ = setup
    final, _ = drive(client, h, start_verify(client, h, target.id), identity=2)
    assert final["final_outcome"] == "NOT_MATCHED" and final["final_distance"] > 0.68


def test_no_face_and_multiple_faces_are_non_terminal(client, setup):
    h, target, _ = setup
    s = start_verify(client, h, target.id)
    r = submit(client, h, s, frame(face_count=0)).json()
    assert r["frame"]["outcome"] == "NO_FACE" and r["status"] == "ACTIVE"
    r = submit(client, h, r, frame(face_count=2)).json()
    assert r["frame"]["outcome"] == "MULTIPLE_FACES" and "Only one person" in r["frame"]["guidance"][0]
    assert r["status"] == "ACTIVE"


def test_spoof_fails_liveness(client, setup):
    h, target, _ = setup
    final, _ = drive(client, h, start_verify(client, h, target.id),
                     frame_fn=lambda st, ph: frame_for_step(st["challenge"]["type"], 1, ph, real=False))
    assert final["final_outcome"] == "LIVENESS_FAILED"
    ev = client.get("/api/v1/security/events?event_type=LIVENESS_FAILED", headers=admin_headers(client)).json()
    assert ev["total"] >= 1


def test_static_photo_cannot_pass_active_challenge(client, setup):
    """A replayed still (no head movement / blink) never completes the challenge."""
    h, target, _ = setup
    s = start_verify(client, h, target.id)
    steps = s["challenge"]["steps"]
    final, hist = drive(client, h, s, frame_fn=lambda st, ph: frame(identity=1), max_frames=60)
    assert final["status"] == "ACTIVE" or final["final_outcome"] in ("LIVENESS_FAILED", "ATTEMPTS_EXHAUSTED")
    assert all(x["frame"]["outcome"] != "VERIFIED" for x in hist)
    assert steps[1] in ("TURN_LEFT", "TURN_RIGHT", "BLINK")


def test_invalid_image_rejected_without_consuming_frame(client, setup):
    h, target, _ = setup
    s = start_verify(client, h, target.id)
    r = submit(client, h, s, b"not an image at all")
    assert r.status_code == 422 and r.json()["code"] == "INVALID_IMAGE"
    r = client.post(f"/api/v1/sessions/{s['session_id']}/frames", headers=h,
                    data={"seq": "1", "frame_token": s["frame_token"]},
                    files={"file": ("x.gif", b"GIF89a" + b"\x00" * 200, "image/gif")})
    assert r.status_code == 422
    assert submit(client, h, s, frame()).status_code == 200   # same token still valid


def test_oversized_frame_rejected(client, setup):
    h, target, _ = setup
    s = start_verify(client, h, target.id)
    r = submit(client, h, s, frame(size=(1280, 960)) + b"\x00" * 700_000)
    assert r.status_code == 413


def test_expired_session(client, setup):
    h, target, _ = setup
    s = start_verify(client, h, target.id)
    db = SessionLocal()
    db.query(VerificationSession).filter(VerificationSession.id == s["session_id"]).update(
        {"expires_at": datetime.utcnow() - timedelta(seconds=1)})
    db.commit()
    db.close()
    r = submit(client, h, s, frame()).json()
    assert r["status"] == "CLOSED" and r["final_outcome"] == "SESSION_EXPIRED" and r["reason_code"] == "SESSION_TIMEOUT"
    r2 = submit(client, h, s, frame())
    assert r2.status_code == 409 and r2.json()["code"] == "SESSION_CLOSED"


def test_inactivity_timeout(client, setup):
    h, target, _ = setup
    s = start_verify(client, h, target.id)
    db = SessionLocal()
    db.query(VerificationSession).filter(VerificationSession.id == s["session_id"]).update(
        {"last_activity_at": datetime.utcnow() - timedelta(seconds=30)})
    db.commit()
    db.close()
    r = client.get(f"/api/v1/sessions/{s['session_id']}", headers=h).json()
    assert r["final_outcome"] == "SESSION_EXPIRED" and r["reason_code"] == "INACTIVITY_TIMEOUT"


def test_maintenance_expires_abandoned_sessions(client, setup):
    from backend.services.maintenance_service import run_maintenance
    h, target, _ = setup
    s = start_verify(client, h, target.id)
    db = SessionLocal()
    db.query(VerificationSession).filter(VerificationSession.id == s["session_id"]).update(
        {"expires_at": datetime.utcnow() - timedelta(seconds=1)})
    db.commit()
    db.close()
    assert run_maintenance()["sessions_expired"] == 1
    assert _session_row(s["session_id"]).final_outcome == "SESSION_EXPIRED"


def test_other_users_session_is_not_accessible(client, setup):
    h, target, _ = setup
    s = start_verify(client, h, target.id)
    other = make_user(RoleEnum.OPERATOR)
    oh = login(client, other.email)
    assert submit(client, oh, s, frame()).status_code == 404
    assert client.post(f"/api/v1/sessions/{s['session_id']}/cancel", headers=oh).status_code == 404
    assert client.get(f"/api/v1/sessions/{s['session_id']}", headers=oh).status_code == 404
    alerts = client.get("/api/v1/security/alerts", headers=admin_headers(client)).json()["alerts"]
    viol = [a for a in alerts if a["rule"] == "SESSION_ACCESS_VIOLATION"]
    assert viol and viol[0]["subject_id"] == other.id and viol[0]["occurrence_count"] >= 3


def test_stale_token_and_sequence_rejected(client, setup):
    h, target, _ = setup
    s = start_verify(client, h, target.id)
    r1 = submit(client, h, s, frame())
    assert r1.status_code == 200
    r2 = submit(client, h, s, frame())          # old token + old seq
    assert r2.status_code == 409 and r2.json()["code"] == "STALE_FRAME"
    s2 = dict(r1.json())
    s2["frame_seq"] += 5
    assert submit(client, h, s2, frame()).status_code == 409


def test_replayed_frame_bytes_rejected_across_sessions(client, setup):
    h, target, _ = setup
    data = frame()
    s = start_verify(client, h, target.id)
    assert submit(client, h, s, data).status_code == 200
    s2 = start_verify(client, h, target.id)
    r = submit(client, h, s2, data)
    assert r.status_code == 409 and r.json()["code"] == "FRAME_REPLAYED"


def test_frozen_feed_ends_session_as_liveness_failure(client, setup):
    h, target, _ = setup
    data = frame()
    s = start_verify(client, h, target.id)
    assert submit(client, h, s, data).status_code == 200
    s = client.post(f"/api/v1/sessions/{s['session_id']}/resync", headers=h).json()
    for _ in range(4):
        assert submit(client, h, s, data).json()["code"] == "FRAME_REPLAYED"
    final = submit(client, h, s, data).json()
    assert final["status"] == "CLOSED" and final["final_outcome"] == "LIVENESS_FAILED" and final["reason_code"] == "FRAME_REPLAY"


def test_frame_events_record_the_evaluated_step(client, setup):
    h, target, _ = setup
    final, _ = drive(client, h, start_verify(client, h, target.id), identity=1)
    db = SessionLocal()
    events = (db.query(VerificationEvent).filter(VerificationEvent.session_id == final["session_id"],
                                                 VerificationEvent.frame_seq.isnot(None))
              .order_by(VerificationEvent.frame_seq).all())
    db.close()
    assert events[0].step_type == "LOOK_STRAIGHT" and events[1].step_type == "LOOK_STRAIGHT"
    assert events[1].explanation == "Look straight passed"


def test_new_session_supersedes_previous_one(client, setup):
    h, target, _ = setup
    s1 = start_verify(client, h, target.id)
    start_verify(client, h, target.id)
    row = _session_row(s1["session_id"])
    assert not row.is_active and row.final_outcome == "SESSION_CANCELLED" and row.reason_code == "SUPERSEDED"


def test_cancel_session_and_closed_session_rejects_frames(client, setup):
    h, target, _ = setup
    s = start_verify(client, h, target.id)
    r = client.post(f"/api/v1/sessions/{s['session_id']}/cancel", headers=h).json()
    assert r["final_outcome"] == "SESSION_CANCELLED"
    r = submit(client, h, s, frame())
    assert r.status_code == 409 and r.json()["code"] == "SESSION_CLOSED"


def test_completed_session_accepts_no_more_frames(client, setup):
    h, target, _ = setup
    s = start_verify(client, h, target.id)
    token, seq = s["frame_token"], s["frame_seq"]
    final, _ = drive(client, h, s, identity=1)
    assert final["final_outcome"] == "VERIFIED"
    r = submit(client, h, {"session_id": s["session_id"], "frame_seq": final["frame_seq"], "frame_token": token}, frame())
    assert r.status_code == 409


def test_missing_enrolment(client):
    h = admin_headers(client)
    u = make_user(RoleEnum.OPERATOR)
    r = client.post("/api/v1/sessions", json={"target_user_id": u.id}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "NOT_ENROLLED"
    assert client.post("/api/v1/sessions", json={"target_user_id": "nope"}, headers=h).status_code == 404


def test_revoked_enrolment_blocks_verification(client, setup):
    h, target, _ = setup
    ah = admin_headers(client)
    assert client.request("DELETE", f"/api/v1/enrolments/{target.id}", json={"reason": "test"}, headers=ah).status_code == 200
    r = client.post("/api/v1/sessions", json={"target_user_id": target.id}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "ENROLMENT_REVOKED"


def test_withdrawn_consent_blocks_verification(client, setup):
    h, target, _ = setup
    ah = admin_headers(client)
    assert client.post(f"/api/v1/enrolments/{target.id}/consent/withdraw", json={}, headers=ah).status_code == 200
    r = client.post("/api/v1/sessions", json={"target_user_id": target.id}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "CONSENT_WITHDRAWN"
    db = SessionLocal()
    enr = db.query(BiometricEnrolment).filter(BiometricEnrolment.user_id == target.id).one()
    db.close()
    assert enr.embedding_encrypted is None


def test_legacy_template_requires_reenrolment(client, setup):
    h, target, _ = setup
    db = SessionLocal()
    db.query(BiometricEnrolment).filter(BiometricEnrolment.user_id == target.id).update(
        {"template_version": 1, "status": "REENROLMENT_REQUIRED"})
    db.commit()
    db.close()
    r = client.post("/api/v1/sessions", json={"target_user_id": target.id}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "REENROLMENT_REQUIRED"


def test_embedding_dimension_mismatch_is_error_not_match(client, setup):
    import numpy as np
    from backend.services.biometric.templates import encrypt_vectors
    h, target, _ = setup
    db = SessionLocal()
    db.query(BiometricEnrolment).filter(BiometricEnrolment.user_id == target.id).update(
        {"embedding_encrypted": encrypt_vectors([np.ones(128)], "ArcFace")})
    db.commit()
    db.close()
    final, _ = drive(client, h, start_verify(client, h, target.id), identity=1)
    assert final["final_outcome"] == "ERROR" and final["reason_code"] == "TEMPLATE_UNUSABLE"


def test_missing_encryption_key_fails_safely(client, setup, monkeypatch):
    from backend.config import settings
    from backend.services.biometric.templates import reset_crypto_cache
    h, target, _ = setup
    monkeypatch.setattr(settings, "embedding_encryption_key", "")
    reset_crypto_cache()
    r = client.post("/api/v1/sessions", json={"target_user_id": target.id}, headers=h)
    assert r.status_code == 503 and r.json()["code"] == "ENCRYPTION_KEY_MISSING"


def test_model_failure_is_error_and_never_a_match(client, setup, monkeypatch):
    h, target, _ = setup
    s = start_verify(client, h, target.id)

    def boom(*a, **k):
        raise RuntimeError("model crashed")
    monkeypatch.setattr(runtime.provider, "analyze", boom)
    outcomes = []
    for _ in range(3):
        r = submit(client, h, s, frame())
        assert r.status_code == 200
        s = r.json()
        outcomes.append(s["frame"]["outcome"])
    assert outcomes == ["ERROR", "ERROR", "ERROR"]
    assert s["status"] == "CLOSED" and s["final_outcome"] == "ERROR"


def test_inference_busy_returns_503_and_releases_claim(client, setup, monkeypatch):
    h, target, _ = setup
    s = start_verify(client, h, target.id)
    real_run = runtime.run

    def busy(*a, **k):
        raise InferenceBusy("full")
    monkeypatch.setattr(runtime, "run", busy)
    r = submit(client, h, s, frame(nonce=777))
    assert r.status_code == 503 and r.headers.get("Retry-After") == "1"
    monkeypatch.setattr(runtime, "run", real_run)
    r = submit(client, h, s, frame(nonce=777))    # same token, seq and bytes succeed once capacity frees
    assert r.status_code == 200


def test_database_failure_does_not_verify(client, setup, monkeypatch):
    h, target, _ = setup
    s = start_verify(client, h, target.id)

    def broken(*a, **k):
        raise OperationalError("SELECT", {}, Exception("database is locked"))
    monkeypatch.setattr(capture_service, "_reference_blob", broken)
    r = submit(client, h, s, frame())
    assert r.status_code == 500
    assert "database is locked" not in r.text
    row = _session_row(s["session_id"])
    assert row.final_outcome is None and row.in_flight is False   # claim released, session not wedged


def test_concurrent_submissions_only_one_wins(client, setup, monkeypatch):
    h, target, _ = setup
    s = start_verify(client, h, target.id)
    gate = threading.Event()
    real_analyze = runtime.provider.analyze

    def slow(bgr):
        gate.wait(2)
        return real_analyze(bgr)
    monkeypatch.setattr(runtime.provider, "analyze", slow)
    results: list[int] = []

    def go(n):
        results.append(submit(client, h, s, frame(nonce=9000 + n)).status_code)
    threads = [threading.Thread(target=go, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    import time
    time.sleep(0.5)
    gate.set()
    for t in threads:
        t.join(10)
    assert sorted(results).count(200) == 1, results
    assert all(code == 409 for code in results if code != 200)
    assert _session_row(s["session_id"]).frame_count == 1


def test_viewer_cannot_create_sessions(client, setup):
    _, target, _ = setup
    v = make_user(RoleEnum.VIEWER)
    r = client.post("/api/v1/sessions", json={"target_user_id": target.id}, headers=login(client, v.email))
    assert r.status_code == 403


def test_operator_response_includes_scores_and_guidance(client, setup):
    h, target, _ = setup
    s = start_verify(client, h, target.id)
    assert s["capture"]["interval_ms"] == 350 and s["frame_token"]
    r = submit(client, h, s, frame(width_code=1)).json()
    assert r["frame"]["outcome"] == "QUALITY_FAILED" and "Move closer" in r["frame"]["guidance"]
    assert r["frame"]["face_box"] is not None
