"""Network recovery: a lost response leaves the client with a stale token; resync restores it."""
from __future__ import annotations

from backend.models.user import RoleEnum
from backend.tests.helpers import drive, enrol_direct, frame, login, make_user, start_verify, submit


def test_resync_after_lost_response(client):
    op = make_user(RoleEnum.OPERATOR)
    t = make_user(RoleEnum.OPERATOR)
    enrol_direct(t.id)
    h = login(client, op.email)
    s = start_verify(client, h, t.id)
    assert submit(client, h, s, frame()).status_code == 200     # response "lost": client keeps old token
    r = submit(client, h, s, frame())
    assert r.status_code == 409 and r.json()["code"] == "STALE_FRAME"
    fresh = client.post(f"/api/v1/sessions/{s['session_id']}/resync", headers=h).json()
    assert fresh["frame_token"] != s["frame_token"] and fresh["frame_seq"] == 1
    final, _ = drive(client, h, fresh, identity=1)
    assert final["final_outcome"] == "VERIFIED"


def test_resync_is_owner_only(client):
    op = make_user(RoleEnum.OPERATOR)
    t = make_user(RoleEnum.OPERATOR)
    enrol_direct(t.id)
    s = start_verify(client, login(client, op.email), t.id)
    other = make_user(RoleEnum.OPERATOR)
    assert client.post(f"/api/v1/sessions/{s['session_id']}/resync", headers=login(client, other.email)).status_code == 404
