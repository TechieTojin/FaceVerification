"""Talking to the central server: provisioning, signed device requests, bundle decryption."""
from __future__ import annotations

import hashlib
import json
import secrets
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from jose import jwt

from kiosk.store import Identity, b64, unb64

BUNDLE_INFO = b"fv-offline-bundle-v1"
TIMEOUT_S = 10


class ServerUnreachable(RuntimeError):
    pass


class ServerError(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def canonical(method: str, path: str, ts: str, nonce: str, body: bytes) -> bytes:
    """Must match backend.services.device_service.canonical."""
    return f"{method.upper()}\n{path}\n{ts}\n{nonce}\n{hashlib.sha256(body).hexdigest()}".encode()


def _request(url: str, method: str, body: Optional[bytes], headers: dict[str, str]) -> dict[str, Any]:
    req = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        try:
            data = json.loads(exc.read() or b"{}")
        except ValueError:
            data = {}
        raise ServerError(exc.code, data.get("code", "HTTP_ERROR"), data.get("detail", f"HTTP {exc.code}")) from None
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ServerUnreachable(str(exc)) from exc


def public_pem(key: ec.EllipticCurvePrivateKey) -> str:
    return key.public_key().public_bytes(serialization.Encoding.PEM,
                                         serialization.PublicFormat.SubjectPublicKeyInfo).decode()


def provision(server_url: str, code: str, key: ec.EllipticCurvePrivateKey, software_version: str, platform: str,
              key_protection: str) -> dict[str, Any]:
    body = json.dumps({"code": code, "public_key_pem": public_pem(key), "software_version": software_version,
                       "platform": platform, "key_protection": key_protection}).encode()
    return _request(server_url.rstrip("/") + "/api/v1/device-api/provision", "POST", body, {})


def signed(ident: Identity, method: str, path: str, payload: Optional[dict] = None) -> dict[str, Any]:
    body = json.dumps(payload or {}).encode()
    ts, nonce = str(int(time.time())), secrets.token_hex(16)
    sig = ident.private_key.sign(canonical(method, path, ts, nonce, body), ec.ECDSA(hashes.SHA256()))
    return _request(ident.server_url.rstrip("/") + path, method, body, {
        "X-Device-Id": ident.device_id, "X-Device-Timestamp": ts, "X-Device-Nonce": nonce,
        "X-Device-Signature": b64(sig)})


def fetch_jwks(server_url: str) -> dict[str, Any]:
    return _request(server_url.rstrip("/") + "/api/v1/integrations/.well-known/jwks.json", "GET", None, {})


def open_bundle(ident: Identity, resp: dict[str, Any], jwks: dict[str, Any]) -> dict[str, Any]:
    """Verify the server's signature over the ciphertext, then decrypt with the device key."""
    ct = unb64(resp["ciphertext"])
    header = jwt.get_unverified_header(resp["signature"])
    jwk = next((k for k in jwks.get("keys", []) if k.get("kid") == header.get("kid")), None)
    if jwk is None:
        raise ValueError("Bundle signed with an unknown server key")
    claims = jwt.decode(resp["signature"], jwk, algorithms=["ES256"])
    if (claims.get("sha256") != hashlib.sha256(ct).hexdigest() or claims.get("device_id") != ident.device_id
            or claims.get("bundle_id") != resp["bundle_id"]):
        raise ValueError("Bundle signature does not match its content")
    epk = serialization.load_pem_public_key(resp["epk"].encode())
    shared = ident.private_key.exchange(ec.ECDH(), epk)
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=resp["bundle_id"].encode(), info=BUNDLE_INFO).derive(shared)
    plain = AESGCM(key).decrypt(unb64(resp["nonce"]), ct, f"{resp['bundle_id']}|{ident.device_id}".encode())
    bundle = json.loads(plain)
    if bundle.get("device_id") != ident.device_id or bundle.get("bundle_id") != resp["bundle_id"]:
        raise ValueError("Bundle was issued for a different device")
    return bundle
