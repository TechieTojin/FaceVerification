"""QR pairing and mobile-to-desktop verification."""
from __future__ import annotations

import threading
from datetime import datetime, timedelta

from backend.core import events
from backend.database import SessionLocal
from backend.models.pairing import CapturePairing
from backend.models.user import RoleEnum
from backend.services import pairing_service
from backend.services.capture_service import CaptureError
from backend.tests.helpers import admin_headers, drive, enrol_direct, login, make_org, make_user, start_verify

P = "/api/v1/pairings"
R = "/api/v1/remote"


def _setup(client):
    op = make_user(RoleEnum.OPERATOR, name="Desk Operator")
    subject = make_user(RoleEnum.VIEWER, name="Mobile Subject Person")
    enrol_direct(subject.id, identity=5)
    return op, subject, login(client, op.email)


def _token(p: dict) -> str:
    return p["qr_url"].rsplit("/m/", 1)[1]


def _code(p_id: str, raw_code: str) -> str:
    return raw_code


def _claim(client, token):
    r = client.post(f"{R}/link/{token}/claim", json={"accept": True})
    assert r.status_code == 200, r.text
    return r.json()


def test_full_qr_flow_reaches_one_final_decision(client, monkeypatch):
    published = []
    real_publish = events.bus.publish
    monkeypatch.setattr(events.bus, "publish", lambda ch, ev, data, org_id=None, user_id=None: (
        published.append((ch, ev, user_id, data)), real_publish(ch, ev, data, org_id=org_id, user_id=user_id)))
    op, subject, h = _setup(client)
    r = client.post(P, json={"target_user_id": subject.id}, headers=h)
    assert r.status_code == 201, r.text
    pairing = r.json()
    assert pairing["status"] == "WAITING" and "/m/" in pairing["qr_url"]
    token = _token(pairing)
    assert subject.id not in pairing["qr_url"] and "Mobile" not in pairing["qr_url"]   # opaque reference only

    info = client.get(f"{R}/link/{token}").json()
    assert info["subject"] == "Mobile P." and info["requested_by"] == "Desk Operator"
    claimed = _claim(client, token)
    secret, code = claimed["pairing_secret"], claimed["confirmation_code"]
    assert claimed["state"]["status"] == "AWAITING_CONFIRMATION" and len(code) == 6
    ph = {"X-Pairing-Secret": secret}
    # Phone cannot capture before the operator confirms the code.
    assert client.post(f"{R}/session", headers=ph).json()["code"] == "AWAITING_CONFIRMATION"
    # Second claim of the same QR fails.
    r = client.post(f"{R}/link/{token}/claim", json={"accept": True})
    assert r.status_code == 410 and r.json()["code"] == "PAIRING_USED"

    wrong = "000000" if code != "000000" else "111111"
    assert client.post(f"{P}/{pairing['id']}/confirm", json={"code": wrong}, headers=h).status_code == 400
    r = client.post(f"{P}/{pairing['id']}/confirm", json={"code": code}, headers=h)
    assert r.status_code == 200 and r.json()["status"] == "READY"

    sess = client.post(f"{R}/session", headers=ph)
    assert sess.status_code == 201, sess.text
    sess = sess.json()
    assert sess["channel"] == "MOBILE"
    final, history = drive(client, ph, sess, identity=5)
    assert final["final_outcome"] == "VERIFIED"
    assert all("distance" not in (s["frame"] or {}) for s in history)          # phone never sees scores

    desk = client.get(f"{P}/{pairing['id']}", headers=h).json()
    assert desk["status"] == "COMPLETED" and desk["result_outcome"] == "VERIFIED" and desk["device"]
    phone = client.get(f"{R}/state", headers=ph).json()
    assert phone["status"] == "COMPLETED" and phone["result_outcome"] == "VERIFIED"
    assert "requested_by" not in phone
    r = client.post(f"{R}/session", headers=ph)
    assert r.status_code == 409 and r.json()["code"] == "PAIRING_COMPLETED"

    names = [ev for ch, ev, uid, _ in published if ch == "pairing" and uid == op.id]
    for ev in ("pairing_claimed", "pairing_confirmed", "pairing_session_started", "pairing_progress", "pairing_result"):
        assert ev in names, (ev, names)
    assert all(uid == op.id for ch, ev, uid, _ in published if ch == "pairing")
    hist = client.get(f"/api/v1/history/sessions/{sess['session_id']}", headers=admin_headers(client)).json()
    assert hist["channel"] == "MOBILE" and hist["operator_user_id"] == op.id


def test_expired_reused_invalid_and_cancelled_links(client):
    op, subject, h = _setup(client)
    assert client.get(f"{R}/link/not-a-real-token").status_code == 404
    p1 = client.post(P, json={"target_user_id": subject.id}, headers=h).json()
    db = SessionLocal()
    db.query(CapturePairing).filter(CapturePairing.id == p1["id"]).update(
        {"expires_at": datetime.utcnow() - timedelta(seconds=1)})
    db.commit()
    db.close()
    r = client.get(f"{R}/link/{_token(p1)}")
    assert r.status_code == 410 and r.json()["code"] == "PAIRING_EXPIRED"
    assert client.post(f"{R}/link/{_token(p1)}/claim", json={"accept": True}).status_code == 410
    assert client.get(f"{P}/{p1['id']}", headers=h).json()["status"] == "EXPIRED"
    notes = client.get("/api/v1/notifications", headers=h).json()["notifications"]
    assert any(n["type"] == "PAIRING_EXPIRED" for n in notes)

    # Regenerate gives a new QR; the old token stays dead.
    p2 = client.post(f"{P}/{p1['id']}/regenerate", headers=h).json()
    assert p2["id"] != p1["id"] and client.get(f"{R}/link/{_token(p2)}").status_code == 200

    # Operator cancellation propagates to the phone and stops capture.
    claimed = _claim(client, _token(p2))
    ph = {"X-Pairing-Secret": claimed["pairing_secret"]}
    client.post(f"{P}/{p2['id']}/confirm", json={"code": claimed["confirmation_code"]}, headers=h)
    sess = client.post(f"{R}/session", headers=ph).json()
    assert client.post(f"{P}/{p2['id']}/cancel", headers=h).json()["status"] == "CANCELLED"
    assert client.get(f"{R}/state", headers=ph).json()["status"] == "CANCELLED"
    assert client.get(f"/api/v1/sessions/{sess['session_id']}", headers=ph).status_code == 403   # pairing closed
    s = client.get(f"/api/v1/sessions/{sess['session_id']}", headers=h)
    assert s.status_code == 404   # operator does not own the phone's session either
    assert client.post(f"{R}/session", headers=ph).status_code == 410

    # Phone-side cancellation is visible on the desktop.
    p3 = client.post(P, json={"target_user_id": subject.id}, headers=h).json()
    c3 = _claim(client, _token(p3))
    client.post(f"{R}/cancel", headers={"X-Pairing-Secret": c3["pairing_secret"]})
    assert client.get(f"{P}/{p3['id']}", headers=h).json()["cancelled_reason"] == "PHONE_CANCELLED"


def test_wrong_codes_cancel_pairing_and_raise_alert(client):
    op, subject, h = _setup(client)
    p = client.post(P, json={"target_user_id": subject.id}, headers=h).json()
    c = _claim(client, _token(p))
    bad = "999999" if c["confirmation_code"] != "999999" else "888888"
    for _ in range(2):
        assert client.post(f"{P}/{p['id']}/confirm", json={"code": bad}, headers=h).status_code == 400
    r = client.post(f"{P}/{p['id']}/confirm", json={"code": bad}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "PAIRING_CANCELLED"
    # Even the right code no longer works.
    assert client.post(f"{P}/{p['id']}/confirm", json={"code": c["confirmation_code"]}, headers=h).status_code == 409
    alerts = client.get("/api/v1/security/alerts", headers=admin_headers(client)).json()["alerts"]
    assert any(a["rule"] == "PAIRING_ABUSE" for a in alerts)


def test_simultaneous_claims_only_one_wins(client):
    op, subject, h = _setup(client)
    token = _token(client.post(P, json={"target_user_id": subject.id}, headers=h).json())
    results, barrier = [], threading.Barrier(4)

    def worker(i):
        db = SessionLocal()
        try:
            barrier.wait()
            pairing_service.claim(db, token, f"10.0.0.{i}", "UA")
            results.append("ok")
        except Exception as exc:  # noqa: BLE001
            results.append(getattr(exc, "code", type(exc).__name__))
        finally:
            db.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 1 and all(r == "PAIRING_USED" for r in results if r != "ok"), results


def test_pairings_are_isolated_between_operators_and_organisations(client):
    op, subject, h = _setup(client)
    p = client.post(P, json={"target_user_id": subject.id}, headers=h).json()
    other = make_user(RoleEnum.OPERATOR)
    ho = login(client, other.email)
    assert client.get(f"{P}/{p['id']}", headers=ho).status_code == 404
    assert client.post(f"{P}/{p['id']}/cancel", headers=ho).status_code == 404
    org_b = make_org("Pairing B", "pairing-b")
    b_admin = make_user(RoleEnum.ADMIN, org_id=org_b)
    hb = login(client, b_admin.email)
    assert client.get(f"{P}/{p['id']}", headers=hb).status_code == 404
    # Org B cannot create a QR for an org A subject.
    assert client.post(P, json={"target_user_id": subject.id}, headers=hb).status_code == 404
    # Viewers cannot create pairings at all.
    hv = login(client, subject.email)
    assert client.post(P, json={"target_user_id": subject.id}, headers=hv).status_code == 403

    # A pairing secret drives only its own session.
    c = _claim(client, _token(p))
    client.post(f"{P}/{p['id']}/confirm", json={"code": c["confirmation_code"]}, headers=h)
    ph = {"X-Pairing-Secret": c["pairing_secret"]}
    ops_session = start_verify(client, h, subject.id)
    assert client.get(f"/api/v1/sessions/{ops_session['session_id']}", headers=ph).status_code == 404
    assert client.get(f"{R}/state", headers={"X-Pairing-Secret": "forged-secret"}).status_code == 401


def test_unenrolled_subject_cannot_be_paired(client):
    op = make_user(RoleEnum.OPERATOR)
    nobody = make_user(RoleEnum.VIEWER)
    r = client.post(P, json={"target_user_id": nobody.id}, headers=login(client, op.email))
    assert r.status_code == 409 and r.json()["code"] == "NOT_ENROLLED"
