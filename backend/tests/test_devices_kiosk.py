"""Device registration, signed device requests, cameras, and the offline kiosk (provision → offline → sync)."""
from __future__ import annotations

import base64
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from backend.config import settings
from backend.database import SessionLocal
from backend.models.devices import Device
from backend.models.user import RoleEnum
from backend.tests.helpers import admin_headers, enrol_direct, frame_for_step, login, make_org, make_user
from backend.tests.test_passkeys_mfa import step_up_password
from kiosk import client as kclient, keystore
from kiosk.store import Identity, KioskStore, b64

D, DA, C = "/api/v1/devices", "/api/v1/device-api", "/api/v1/cameras"


class Transport:
    """Routes the kiosk's HTTP calls into the test server; `online=False` simulates a network outage."""

    def __init__(self, client):
        self.client, self.online = client, True

    def __call__(self, url, method, body, headers):
        if not self.online:
            raise kclient.ServerUnreachable("network down (simulated)")
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        self.client.cookies.clear()
        r = self.client.request(method, path, content=body, headers={"Content-Type": "application/json", **headers})
        data = r.json() if r.content else {}
        if r.status_code >= 400:
            raise kclient.ServerError(r.status_code, data.get("code", ""), data.get("detail", ""))
        return data


@pytest.fixture
def transport(client, monkeypatch):
    t = Transport(client)
    monkeypatch.setattr(kclient, "_request", t)
    monkeypatch.setenv("KIOSK_PASSPHRASE", "test-only-kiosk-passphrase")
    return t


def _admin(client):
    return step_up_password(client, admin_headers(client), settings.initial_admin_password)


def _new_device(client, hs, **kw):
    r = client.post(D, json={"name": "Lobby kiosk", "kind": "KIOSK", "offline_enabled": True,
                             "offline_validity_hours": 24, **kw}, headers=hs)
    assert r.status_code == 201, r.text
    return r.json()


def _signed(client, device_id, key, method, path, payload=None, ts=None, nonce=None, sign_with=None):
    body = json.dumps(payload or {}).encode()
    ts = ts or str(int(time.time()))
    nonce = nonce or base64.b16encode(__import__("os").urandom(16)).decode()
    sig = (sign_with or key).sign(kclient.canonical(method, path, ts, nonce, body), ec.ECDSA(hashes.SHA256()))
    client.cookies.clear()
    return client.request(method, path, content=body, headers={
        "Content-Type": "application/json", "X-Device-Id": device_id, "X-Device-Timestamp": ts,
        "X-Device-Nonce": nonce, "X-Device-Signature": b64(sig)})


def test_device_provisioning_signatures_replay_and_revocation(client):
    hs = _admin(client)
    assert client.post(D, json={"name": "x", "kind": "KIOSK"}, headers=admin_headers(client)).json()["code"] == "STEP_UP_REQUIRED"
    dev = _new_device(client, hs)
    assert dev["status"] == "PENDING" and len(dev["provisioning_code"]) == 12
    key = ec.generate_private_key(ec.SECP256R1())
    pem = kclient.public_pem(key)
    assert client.post(f"{DA}/provision", json={"code": "WRONGCODE123", "public_key_pem": pem}).status_code == 401
    r = client.post(f"{DA}/provision", json={"code": dev["provisioning_code"], "public_key_pem": pem,
                                              "key_protection": "software"})
    assert r.status_code == 201 and r.json()["device_id"] == dev["id"]
    assert client.post(f"{DA}/provision", json={"code": dev["provisioning_code"], "public_key_pem": pem}).status_code == 401

    assert _signed(client, dev["id"], key, "GET", f"{DA}/whoami").json()["status"] == "ACTIVE"
    # Signature by another key, a reused nonce and a stale timestamp are all refused.
    other = ec.generate_private_key(ec.SECP256R1())
    assert _signed(client, dev["id"], key, "GET", f"{DA}/whoami", sign_with=other).status_code == 401
    r1 = _signed(client, dev["id"], key, "POST", f"{DA}/heartbeat", {"software_version": "4.0.0"}, nonce="a" * 32)
    assert r1.status_code == 200
    r2 = _signed(client, dev["id"], key, "POST", f"{DA}/heartbeat", {"software_version": "4.0.0"}, nonce="a" * 32)
    assert r2.status_code == 401 and r2.json()["code"] == "REPLAYED_REQUEST"
    r3 = _signed(client, dev["id"], key, "GET", f"{DA}/whoami", ts=str(int(time.time()) - 3600))
    assert r3.json()["code"] == "CLOCK_SKEW"
    ha = admin_headers(client)
    view = client.get(f"{D}/{dev['id']}", headers=ha).json()
    assert view["connection"] == "ONLINE" and view["software_version"] == "4.0.0" and view["hardware_attested"] is False
    alerts = client.get("/api/v1/security/alerts", headers=ha).json()["alerts"]
    assert any(a["rule"] == "DEVICE_SIGNATURE_INVALID" for a in alerts)

    # Stale heartbeat → OFFLINE.
    db = SessionLocal()
    db.query(Device).filter(Device.id == dev["id"]).update({"last_seen_at": datetime.utcnow() - timedelta(minutes=10)})
    db.commit()
    db.close()
    assert client.get(f"{D}/{dev['id']}", headers=ha).json()["connection"] == "OFFLINE"

    assert client.post(f"{D}/{dev['id']}/revoke", json={"reason": "stolen"}, headers=ha).json()["status"] == "REVOKED"
    r = _signed(client, dev["id"], key, "GET", f"{DA}/whoami")
    assert r.status_code == 403 and r.json()["code"] == "DEVICE_REVOKED"
    assert any(a["rule"] == "REVOKED_DEVICE_ACCESS"
               for a in client.get("/api/v1/security/alerts", headers=ha).json()["alerts"])
    notes = client.get("/api/v1/notifications", headers=ha).json()["notifications"]
    assert {"DEVICE_REGISTERED", "DEVICE_REVOKED"} <= {n["type"] for n in notes}


def test_device_and_camera_permissions_and_tenant_isolation(client):
    hs = _admin(client)
    dev = _new_device(client, hs, offline_enabled=False)
    op = make_user(RoleEnum.OPERATOR)
    ho = login(client, op.email)
    assert client.get(D, headers=ho).status_code == 200                     # operators may view devices
    assert client.post(f"{D}/{dev['id']}/revoke", json={}, headers=ho).status_code == 403
    viewer = make_user(RoleEnum.VIEWER)
    assert client.get(D, headers=login(client, viewer.email)).status_code == 403
    org_b = make_org("Devices B", "devices-b")
    hb = login(client, make_user(RoleEnum.ADMIN, org_id=org_b).email)
    assert client.get(f"{D}/{dev['id']}", headers=hb).status_code == 404
    assert client.get(D, headers=hb).json()["devices"] == []

    cam = client.post(C, json={"label": "Front desk", "location": "Lobby", "hardware_label": "USB Camera",
                               "device_id": dev["id"]}, headers=hs)
    assert cam.status_code == 201
    cam = cam.json()
    assert cam["connection"] == "NEVER_SEEN"
    assert client.post(C, json={"label": "x"}, headers=ho).status_code == 403
    r = client.post(f"{C}/{cam['id']}/heartbeat", json={"state": "streaming"}, headers=ho)
    assert r.status_code == 200 and r.json()["connection"] == "ONLINE"
    assert client.patch(f"{C}/{cam['id']}", json={"label": "Front desk 2"}, headers=hs).json()["label"] == "Front desk 2"
    assert client.post(f"{C}/{cam['id']}/heartbeat", json={}, headers=hb).status_code == 404
    assert client.post(C, json={"label": "Theirs", "device_id": dev["id"]}, headers=hb).status_code == 404

    # A session on a registered camera records it; a disabled camera cannot be used.
    subject = make_user(RoleEnum.VIEWER)
    enrol_direct(subject.id, identity=3)
    r = client.post("/api/v1/sessions", json={"target_user_id": subject.id, "camera_id": cam["id"]}, headers=ho)
    assert r.status_code == 201 and r.json()["camera_id"] == cam["id"]
    client.post(f"{C}/{cam['id']}/disable", headers=hs)
    r = client.post("/api/v1/sessions", json={"target_user_id": subject.id, "camera_id": cam["id"]}, headers=ho)
    assert r.status_code == 409 and r.json()["code"] == "CAMERA_DISABLED"
    detail = client.get(f"{C}/{cam['id']}", headers=hs).json()
    assert {a["action"] for a in detail["activity"]} >= {"CAMERA_REGISTERED", "CAMERA_UPDATED", "CAMERA_DISABLED"}
    assert len(detail["recent_sessions"]) == 1


def _provision_kiosk(client, transport, tmp_path, subjects):
    hs = _admin(client)
    dev = _new_device(client, hs)
    store = KioskStore(tmp_path / "kiosk")
    key, data_key = store.create_identity("passphrase")
    info = kclient.provision("http://testserver", dev["provisioning_code"], key, "4.0.0", "test", "passphrase")
    store.save_identity(Identity(device_id=info["device_id"], server_url="http://testserver", protection="passphrase",
                                 data_key=data_key, private_key=key, name=info["name"]))
    r = client.put(f"{D}/{dev['id']}/offline-subjects", json={"user_ids": [s.id for s in subjects]}, headers=hs)
    assert r.status_code == 200, r.text
    from kiosk.agent import KioskAgent
    agent = KioskAgent(store)
    res = agent.sync(True)
    assert res["bundle_refreshed"], res
    return dev, store, agent


def _drive_local(agent, user_id, identity, real=True):
    view = agent.start_session(user_id)
    phase = 0
    while view["status"] == "ACTIVE":
        data = frame_for_step(view["challenge"]["type"], identity, phase, real=real)
        phase += 1
        token = view["frame_token"]
        view = agent.submit_frame(view["session_id"], view["frame_seq"] + 1, token, data)
    return view


def test_offline_kiosk_verifies_locally_and_syncs_idempotently(client, transport, tmp_path):
    alice = make_user(RoleEnum.VIEWER, name="Alice Offline")
    enrol_direct(alice.id, identity=8)
    bob = make_user(RoleEnum.VIEWER, name="Bob Notauthorised")
    enrol_direct(bob.id, identity=9)
    dev, store, agent = _provision_kiosk(client, transport, tmp_path, [alice])
    assert [s["display_name"] for s in store.subjects()] == ["Alice Offline"]   # only authorised subjects

    transport.online = False                                    # network outage
    assert agent.sync(True)["error"] == "SERVER_UNREACHABLE" and agent.status()["mode"] == "OFFLINE"
    final = _drive_local(agent, alice.id, identity=8)
    assert final["final_outcome"] == "VERIFIED" and final["channel"] == "KIOSK_OFFLINE"
    impostor = _drive_local(agent, alice.id, identity=9)
    assert impostor["final_outcome"] == "NOT_MATCHED"
    with pytest.raises(Exception) as exc:
        agent.start_session(bob.id)
    assert getattr(exc.value, "code", "") == "SUBJECT_NOT_PROVISIONED"
    assert store.counts()["pending"] == 2

    transport.online = True                                     # reconnect
    res = agent.sync(False)
    assert res["uploaded"] == 2 and store.counts()["pending"] == 0
    assert agent.sync(False)["uploaded"] == 0
    # Re-sending the same events (e.g. lost acknowledgement) is idempotent.
    events = [e for e in reversed(store.recent_events())]
    replay = _signed(client, dev["id"], store.identity.private_key, "POST", f"{DA}/offline-events",
                     {"events": [{k: v for k, v in e.items() if k not in ("synced", "synced_at")} |
                                 {"prev_hash": "", "chain_hash": ""} for e in events]})
    assert replay.status_code == 200 and len(replay.json()["duplicates"]) == 2 and replay.json()["acked"] == []
    server = client.get(f"{D}/{dev['id']}/offline-events", headers=admin_headers(client)).json()["events"]
    assert [e["outcome"] for e in server] == ["NOT_MATCHED", "VERIFIED"]
    assert all(e["integrity"] == "OK" for e in server) and server[1]["subject"] == "Alice Offline"


def test_offline_expiry_corruption_and_revocation(client, transport, tmp_path):
    alice = make_user(RoleEnum.VIEWER, name="Alice Kiosk")
    enrol_direct(alice.id, identity=8)
    dev, store, agent = _provision_kiosk(client, transport, tmp_path, [alice])

    # Expired offline authorisation is refused (and the refusal is audited).
    b = store.get("bundle")
    store.set("bundle", {**b, "valid_until": (datetime.utcnow() - timedelta(minutes=1)).isoformat() + "Z"})
    with pytest.raises(Exception) as exc:
        agent.start_session(alice.id)
    assert exc.value.code == "OFFLINE_AUTHORIZATION_EXPIRED"
    store.set("bundle", b)

    # Clock rollback is detected.
    store.set("last_seen_time", (datetime.utcnow() + timedelta(hours=2)).isoformat())
    with pytest.raises(Exception) as exc:
        agent.start_session(alice.id)
    assert exc.value.code == "CLOCK_ROLLBACK"
    store.set("last_seen_time", datetime.utcnow().isoformat())

    # A corrupted template is never used.
    con = sqlite3.connect(store.db_path)
    blob = bytearray(con.execute("SELECT template FROM subjects").fetchone()[0])
    blob[5] ^= 0xFF
    con.execute("UPDATE subjects SET template = ?", (bytes(blob),))
    con.commit()
    with pytest.raises(Exception) as exc:
        agent.start_session(alice.id)
    assert exc.value.code == "LOCAL_DATA_CORRUPTED"
    # A tampered audit log is detected locally...
    con.execute("UPDATE events SET body = replace(body, 'LOCAL_DATA_CORRUPTED', 'NOTHING_HAPPENED')")
    con.commit()
    con.close()
    from kiosk.store import LocalDataError
    with pytest.raises(LocalDataError):
        store.verify_events()
    # ...and a forged chain uploaded directly is flagged by the server.
    forged = {"event_id": "6f9b6b5e-0000-4000-8000-000000000001", "seq": 99, "type": "VERIFICATION",
              "occurred_at": datetime.utcnow().isoformat(timespec="seconds"), "subject_user_id": alice.id,
              "outcome": "VERIFIED", "reason_code": None, "payload": {}, "bundle_id": b["bundle_id"],
              "prev_hash": "", "chain_hash": "0" * 64}
    r = _signed(client, dev["id"], store.identity.private_key, "POST", f"{DA}/offline-events", {"events": [forged]})
    assert r.json()["integrity_issues"] == 1
    ha = admin_headers(client)
    assert any(a["rule"] == "OFFLINE_AUDIT_INTEGRITY" for a in client.get("/api/v1/security/alerts", headers=ha).json()["alerts"])

    # Revocation: the kiosk learns of it at the next sync and erases its templates.
    client.post(f"{D}/{dev['id']}/revoke", json={"reason": "decommissioned"}, headers=ha)
    res = agent.sync(True)
    assert res["error"] == "DEVICE_REVOKED" and agent.revoked and store.subjects() == []
    with pytest.raises(Exception) as exc:
        agent.start_session(alice.id)
    assert exc.value.code == "DEVICE_REVOKED"


def test_bundle_excludes_subjects_whose_enrolment_was_revoked(client, transport, tmp_path):
    a = make_user(RoleEnum.VIEWER, name="Keep Me")
    b = make_user(RoleEnum.VIEWER, name="Revoke Me")
    enrol_direct(a.id, identity=8)
    enrol_direct(b.id, identity=9)
    dev, store, agent = _provision_kiosk(client, transport, tmp_path, [a, b])
    assert len(store.subjects()) == 2
    client.request("DELETE", f"/api/v1/enrolments/{b.id}", json={"reason": "left"}, headers=admin_headers(client))
    agent.sync(True)
    assert [s["display_name"] for s in store.subjects()] == ["Keep Me"]
    # Ineligible subjects cannot be authorised in the first place.
    hs = _admin(client)
    r = client.put(f"{D}/{dev['id']}/offline-subjects", json={"user_ids": [b.id]}, headers=hs)
    assert r.status_code == 422


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI is Windows-only")
def test_dpapi_key_protection_round_trip():
    blob = keystore.protect("dpapi", b"secret-data-key")
    assert blob != b"secret-data-key" and keystore.unprotect("dpapi", blob) == b"secret-data-key"
    with pytest.raises(keystore.KeyProtectionError):
        keystore.unprotect("dpapi", blob[:-4] + b"xxxx")
