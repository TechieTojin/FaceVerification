"""Policy validation, versioning, API, and proof that settings change real behaviour."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from backend.database import SessionLocal
from backend.models.user import RoleEnum
from backend.models.verification import VerificationSession
from backend.services import policy_service as P
from backend.tests.helpers import (
    admin_headers, drive, enrol_direct, frame, login, make_user, start_verify, submit,
)


def _put(client, h, changes, confirm=False):
    return client.put("/api/v1/policy", json={"changes": changes, "confirm": confirm}, headers=h)


# ── validation ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("key,value", [
    ("match_threshold", 0.9), ("match_threshold", 0.1), ("match_threshold", "abc"),
    ("max_attempts", 0), ("max_attempts", 2.5), ("session_timeout_seconds", 5),
    ("passive_liveness_required", "maybe"), ("recognition_model", "VGG-Face"), ("nonexistent", 1),
])
def test_invalid_values_rejected(client, key, value):
    r = _put(client, admin_headers(client), {key: value}, confirm=True)
    assert r.status_code == 422, r.text
    assert key in r.json()["detail"]["errors"]


def test_cross_field_rules():
    cur = P.Policy(1, P.DEFAULTS)
    with pytest.raises(P.PolicyValidationError):
        P.plan_update(cur, {"passive_liveness_required": False, "active_liveness_required": False})
    with pytest.raises(P.PolicyValidationError):
        P.plan_update(cur, {"min_face_width_ratio": 0.45, "max_face_width_ratio": 0.42})


def test_weakening_change_requires_confirmation(client):
    h = admin_headers(client)
    r = _put(client, h, {"match_threshold": 0.68, "max_attempts": 5})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "CONFIRMATION_REQUIRED"
    assert "max_attempts" in r.json()["detail"]["sensitive_changes"]
    r = _put(client, h, {"max_attempts": 5}, confirm=True)
    assert r.status_code == 200 and r.json()["version"] == 2


def test_tightening_change_needs_no_confirmation_and_is_versioned(client):
    h = admin_headers(client)
    r = _put(client, h, {"match_threshold": 0.5})
    assert r.status_code == 200 and r.json()["version"] == 2
    versions = client.get("/api/v1/policy/versions", headers=h).json()["versions"]
    assert [v["version"] for v in versions][:2] == [2, 1]
    assert "match_threshold" in versions[0]["change_summary"]
    ev = client.get("/api/v1/security/audit", headers=h).json()["entries"]
    assert any(e["action"] == "POLICY_UPDATED" for e in ev)


def test_non_admin_cannot_change_policy(client):
    op = make_user(RoleEnum.OPERATOR)
    h = login(client, op.email)
    assert client.get("/api/v1/policy", headers=h).status_code == 200
    assert _put(client, h, {"match_threshold": 0.5}).status_code == 403
    viewer = make_user(RoleEnum.VIEWER)
    assert client.get("/api/v1/policy", headers=login(client, viewer.email)).status_code == 403


# ── behaviour ────────────────────────────────────────────────────────────

def _verify(client, identity, target_identity=1, **policy):
    h = admin_headers(client)
    if policy:
        assert _put(client, h, policy, confirm=True).status_code == 200
    op = make_user(RoleEnum.OPERATOR)
    target = make_user(RoleEnum.OPERATOR)
    enrol_direct(target.id, identity=target_identity)
    oh = login(client, op.email)
    final, hist = drive(client, oh, start_verify(client, oh, target.id), identity=identity)
    return final, hist


def test_threshold_setting_changes_decision(client):
    final, _ = _verify(client, identity=15)                       # distance 0.55
    assert final["final_outcome"] == "VERIFIED"
    final, _ = _verify(client, identity=15, match_threshold=0.5)
    assert final["final_outcome"] == "NOT_MATCHED"
    assert final["threshold"] == 0.5


def test_active_liveness_setting_changes_challenge(client):
    h = admin_headers(client)
    op = make_user(RoleEnum.OPERATOR)
    target = make_user(RoleEnum.OPERATOR)
    enrol_direct(target.id)
    oh = login(client, op.email)
    s = start_verify(client, oh, target.id)
    assert len(s["challenge"]["steps"]) == 4  # straight + 2 actions + straight
    assert _put(client, h, {"active_liveness_required": False}, confirm=True).status_code == 200
    s2 = start_verify(client, oh, target.id)
    assert s2["challenge"]["steps"] == ["LOOK_STRAIGHT"]
    assert s2["policy_version"] == 2


def test_challenge_steps_setting(client):
    h = admin_headers(client)
    assert _put(client, h, {"challenge_steps": 4}).status_code == 200
    target = make_user(RoleEnum.OPERATOR)
    enrol_direct(target.id)
    s = start_verify(client, h, target.id)
    assert len(s["challenge"]["steps"]) == 6


def test_session_timeout_setting(client):
    h = admin_headers(client)
    assert _put(client, h, {"session_timeout_seconds": 45}).status_code == 200
    target = make_user(RoleEnum.OPERATOR)
    enrol_direct(target.id)
    s = start_verify(client, h, target.id)
    created = datetime.fromisoformat(s["created_at"])
    expires = datetime.fromisoformat(s["expires_at"])
    assert (expires - created).total_seconds() == pytest.approx(45, abs=1)


def test_max_frames_setting_ends_session(client):
    h = admin_headers(client)
    assert _put(client, h, {"max_frames_per_session": 20}).status_code == 200
    target = make_user(RoleEnum.OPERATOR)
    enrol_direct(target.id)
    s = start_verify(client, h, target.id)
    for _ in range(25):
        r = submit(client, h, s, frame(face_count=0))
        assert r.status_code == 200
        s = r.json()
        if s["status"] == "CLOSED":
            break
    assert s["final_outcome"] == "ATTEMPTS_EXHAUSTED" and s["reason_code"] == "FRAME_LIMIT"
    assert s["frames_processed"] == 20


def test_quality_setting_changes_frame_gate(client):
    h = admin_headers(client)
    target = make_user(RoleEnum.OPERATOR)
    enrol_direct(target.id)
    s = start_verify(client, h, target.id)
    r = submit(client, h, s, frame(quality=0.5)).json()      # composite quality ≈ 0.83
    assert r["frame"]["outcome"] == "IN_PROGRESS"
    assert _put(client, h, {"verify_min_quality": 0.9, "enrol_min_quality": 0.9}).status_code == 200
    s = start_verify(client, h, target.id)
    r = submit(client, h, s, frame(quality=0.5)).json()
    assert r["frame"]["outcome"] == "QUALITY_FAILED"


def test_passive_liveness_setting(client):
    final, _ = _verify(client, identity=1, passive_liveness_required=False, challenge_steps=1)
    assert final["final_outcome"] == "VERIFIED"
    h = admin_headers(client)
    op = make_user(RoleEnum.OPERATOR)
    target = make_user(RoleEnum.OPERATOR)
    enrol_direct(target.id)
    oh = login(client, op.email)
    from backend.tests.helpers import frame_for_step
    final, _ = drive(client, oh, start_verify(client, oh, target.id),
                     frame_fn=lambda st, ph: frame_for_step(st["challenge"]["type"], 1, ph, real=False))
    assert final["final_outcome"] == "VERIFIED"    # spoof label ignored when passive check disabled
    assert _put(client, h, {"passive_liveness_required": True}).status_code == 200
    final, _ = drive(client, oh, start_verify(client, oh, target.id),
                     frame_fn=lambda st, ph: frame_for_step(st["challenge"]["type"], 1, ph, real=False))
    assert final["final_outcome"] == "LIVENESS_FAILED"


def test_in_progress_session_keeps_its_policy_snapshot(client):
    h = admin_headers(client)
    target = make_user(RoleEnum.OPERATOR)
    enrol_direct(target.id, identity=1)
    s = start_verify(client, h, target.id)          # policy v1, threshold 0.68
    assert _put(client, h, {"match_threshold": 0.4}).status_code == 200   # v2
    final, _ = drive(client, h, s, identity=15)      # distance 0.55
    assert final["final_outcome"] == "VERIFIED" and final["policy_version"] == 1


def test_operator_session_rate_limit(client):
    h = admin_headers(client)
    assert _put(client, h, {"operator_session_rate_limit": 5}).status_code == 200
    target = make_user(RoleEnum.OPERATOR)
    enrol_direct(target.id)
    for _ in range(5):
        start_verify(client, h, target.id)
    r = client.post("/api/v1/sessions", json={"target_user_id": target.id}, headers=h)
    assert r.status_code == 429 and r.json()["code"] == "SESSION_RATE_LIMITED"
    alerts = client.get("/api/v1/security/alerts", headers=h).json()["alerts"]
    assert any(a["rule"] == "SESSION_RATE_LIMITED" for a in alerts)
