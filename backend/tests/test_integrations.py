"""Integration API: credentials, hosted sessions, signed assertions, webhooks, idempotency, isolation."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from jose import jwt

from backend.config import settings
from backend.database import SessionLocal
from backend.models.integrations import WebhookDelivery
from backend.models.user import RoleEnum
from backend.services import integration_service
from backend.tests.helpers import admin_headers, drive, enrol_direct, login, make_org, make_user
from backend.tests.test_passkeys_mfa import bearer, step_up_password

I = "/api/v1/integrations"


class _Receiver:
    def __init__(self, status: int = 200):
        self.status = status
        self.requests: list[tuple[dict, bytes]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append((dict(self.headers), body))
                self.send_response(outer.status)
                self.end_headers()

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}/hook"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def receiver(monkeypatch):
    monkeypatch.setattr(settings, "webhook_allow_private_targets", True)
    integration_service.rate_limiter.reset()
    r = _Receiver()
    yield r
    r.close()


def _admin_stepped_up(client):
    return step_up_password(client, admin_headers(client), settings.initial_admin_password)


def _client(client, hs, **kw) -> dict:
    r = client.post(f"{I}/clients", json={"name": "Shop backend", **kw}, headers=hs)
    assert r.status_code == 201, r.text
    return r.json()


def _api(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _complete_hosted(client, url: str, identity: int) -> dict:
    token = url.rsplit("/m/", 1)[1]
    info = client.get(f"/api/v1/remote/link/{token}").json()
    assert info["needs_operator_confirmation"] is False
    claim = client.post(f"/api/v1/remote/link/{token}/claim", json={"accept": True}).json()
    ph = {"X-Pairing-Secret": claim["pairing_secret"]}
    sess = client.post("/api/v1/remote/session", headers=ph).json()
    final, _ = drive(client, ph, sess, identity=identity)
    return final


def test_end_to_end_hosted_verification_assertion_and_webhook(client, receiver):
    hs = _admin_stepped_up(client)
    c = _client(client, hs)
    key = c["api_key"]
    assert key.startswith("fvk_") and key not in json.dumps(client.get(f"{I}/clients/{c['id']}", headers=hs).json())
    hook = client.post(f"{I}/clients/{c['id']}/webhooks", json={"url": receiver.url}, headers=hs).json()
    secret = hook["secret"]
    subject = make_user(RoleEnum.VIEWER, name="Api Subject")
    enrol_direct(subject.id, identity=4)

    r = client.post(f"{I}/sessions", json={"subject_email": subject.email, "external_ref": "order-42"},
                    headers={**_api(key), "Idempotency-Key": "abc-1"})
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["verification_url"] and "embedding" not in json.dumps(created)
    # Idempotent replay returns the same verification; a different body with the same key is refused.
    again = client.post(f"{I}/sessions", json={"subject_email": subject.email, "external_ref": "order-42"},
                        headers={**_api(key), "Idempotency-Key": "abc-1"})
    assert again.status_code == 201 and again.json()["id"] == created["id"] and again.headers["Idempotent-Replayed"]
    clash = client.post(f"{I}/sessions", json={"subject_email": subject.email, "external_ref": "other"},
                        headers={**_api(key), "Idempotency-Key": "abc-1"})
    assert clash.status_code == 409

    final = _complete_hosted(client, created["verification_url"], identity=4)
    assert final["final_outcome"] == "VERIFIED" and final["channel"] == "HOSTED"
    status = client.get(f"{I}/sessions/{created['id']}", headers=_api(key)).json()
    assert status["status"] == "COMPLETED" and status["outcome"] == "VERIFIED"
    token = status["assertion"]

    jwks = client.get(f"{I}/.well-known/jwks.json").json()
    header = jwt.get_unverified_header(token)
    jwk = next(k for k in jwks["keys"] if k["kid"] == header["kid"])
    claims = jwt.decode(token, jwk, algorithms=["ES256"], audience=c["id"], issuer=integration_service.issuer())
    for k in ("iss", "aud", "sub", "outcome", "iat", "exp", "jti", "policy_version"):
        assert k in claims
    assert claims["sub"] == "order-42" and claims["outcome"] == "VERIFIED"
    with pytest.raises(Exception):
        jwt.decode(token, jwk, algorithms=["ES256"], audience="someone-else")

    first = client.post(f"{I}/assertions/introspect", json={"assertion": token}, headers=_api(key)).json()
    assert first["active"] is True
    second = client.post(f"{I}/assertions/introspect", json={"assertion": token}, headers=_api(key)).json()
    assert second == {"active": False, "reason": "already_used", "jti": claims["jti"]}
    tampered = token[:-4] + ("AAAA" if not token.endswith("AAAA") else "BBBB")
    assert client.post(f"{I}/assertions/introspect", json={"assertion": tampered}, headers=_api(key)).json()["active"] is False

    db = SessionLocal()
    integration_service.deliver_due(db)
    db.close()
    assert len(receiver.requests) == 1
    headers, body = receiver.requests[0]
    sig = headers.get("FV-Signature") or headers.get("Fv-Signature")
    assert integration_service.verify_signature(secret, body, sig)
    assert not integration_service.verify_signature(secret, body + b" ", sig)
    assert not integration_service.verify_signature("whsec_wrong", body, sig)
    event = json.loads(body)
    assert event["type"] == "verification.completed" and event["data"]["outcome"] == "VERIFIED"
    assert event["data"]["assertion"]


def test_webhook_failures_retry_with_backoff(client, receiver):
    receiver.status = 500
    hs = _admin_stepped_up(client)
    c = _client(client, hs)
    hook = client.post(f"{I}/clients/{c['id']}/webhooks", json={"url": receiver.url, "events": ["ping"]}, headers=hs).json()
    r = client.post(f"{I}/webhooks/{hook['id']}/test", headers=hs).json()
    assert r["status"] == "PENDING" and r["last_status_code"] == 500 and r["attempts"] == 1
    db = SessionLocal()
    d = db.query(WebhookDelivery).filter(WebhookDelivery.id == r["id"]).one()
    assert d.next_attempt_at > datetime.utcnow() + timedelta(seconds=30)
    for _ in range(5):
        d.next_attempt_at = datetime.utcnow()
        db.commit()
        integration_service.deliver_due(db)
        db.refresh(d)
    assert d.status == "FAILED" and d.attempts == 6
    db.close()
    receiver.status = 204
    r = client.post(f"/api/v1/integrations/deliveries/{r['id']}/retry", headers=hs).json()
    assert r["status"] == "DELIVERED"


def test_credentials_scopes_rotation_revocation_and_rate_limit(client, receiver):
    hs = _admin_stepped_up(client)
    # Creating credentials requires step-up.
    plain = admin_headers(client)
    assert client.post(f"{I}/clients", json={"name": "No step-up"}, headers=plain).json()["code"] == "STEP_UP_REQUIRED"
    c = _client(client, hs, scopes=["sessions:read"], rate_limit_per_minute=3)
    key = c["api_key"]
    assert client.post(f"{I}/sessions", json={"subject_email": "x@test.local"}, headers=_api(key)).json()["code"] == "INSUFFICIENT_SCOPE"
    assert client.get(f"{I}/sessions/nope", headers=_api(key)).status_code == 404
    assert client.get(f"{I}/sessions/nope", headers=_api(key)).status_code == 404
    r = client.get(f"{I}/sessions/nope", headers=_api(key))
    assert r.status_code == 429 and "Retry-After" in r.headers
    integration_service.rate_limiter.reset()

    assert client.get(f"{I}/sessions/nope", headers=_api(key[:-2] + "zz")).status_code == 401
    alerts = client.get("/api/v1/security/alerts", headers=plain).json()["alerts"]
    assert any(a["rule"] == "API_CREDENTIAL_ABUSE" for a in alerts)

    rot = client.post(f"{I}/clients/{c['id']}/credentials/rotate", json={"grace_hours": 0}, headers=hs).json()
    new_key = rot["api_key"]
    assert client.get(f"{I}/sessions/nope", headers=_api(key)).status_code == 401       # old key expired
    assert client.get(f"{I}/sessions/nope", headers=_api(new_key)).status_code == 404
    client.post(f"{I}/clients/{c['id']}/credentials/{rot['credential_id']}/revoke", headers=hs)
    assert client.get(f"{I}/sessions/nope", headers=_api(new_key)).status_code == 401
    # Viewers and operators cannot manage integrations.
    op = make_user(RoleEnum.OPERATOR)
    assert client.get(f"{I}/clients", headers=login(client, op.email)).status_code == 403


def test_tenant_isolation_and_private_webhook_targets(client, receiver, monkeypatch):
    hs = _admin_stepped_up(client)
    key_a = _client(client, hs)["api_key"]
    org_b = make_org("Api B", "api-b")
    b_subject = make_user(RoleEnum.VIEWER, org_id=org_b)
    enrol_direct(b_subject.id, identity=6, org_id=org_b)
    r = client.post(f"{I}/sessions", json={"subject_user_id": b_subject.id}, headers=_api(key_a))
    assert r.status_code == 404 and r.json()["code"] == "SUBJECT_NOT_FOUND"
    b_admin = make_user(RoleEnum.ADMIN, org_id=org_b)
    hb = step_up_password(client, login(client, b_admin.email))
    cb = _client(client, hb)
    sess_b = client.post(f"{I}/sessions", json={"subject_user_id": b_subject.id}, headers=_api(cb["api_key"])).json()
    assert client.get(f"{I}/sessions/{sess_b['id']}", headers=_api(key_a)).status_code == 404
    assert client.get(f"{I}/clients/{cb['id']}", headers=hs).status_code == 404

    monkeypatch.setattr(settings, "webhook_allow_private_targets", False)
    for url in ("http://example.com/hook", "https://127.0.0.1/hook", "https://localhost/hook", "ftp://x/y"):
        r = client.post(f"{I}/clients/{cb['id']}/webhooks", json={"url": url}, headers=hb)
        assert r.status_code == 422, url
