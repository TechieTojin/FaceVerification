"""Small cryptographic helpers shared by the v4 services.

* `keyed_hash` — HMAC-SHA256 keyed with SECRET_KEY, for high-entropy secrets that
  must be looked up but never stored (API secrets, recovery codes, pairing and
  provisioning tokens, OTPs). An attacker with only a database copy cannot
  verify guesses without the server key.
* `seal` / `unseal` — Fernet (AES-128-CBC + HMAC) with EMBEDDING_ENCRYPTION_KEY,
  for secrets the server must read back (TOTP seeds, webhook secrets, signing keys).
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

from cryptography.fernet import Fernet

from backend.config import settings


def keyed_hash(value: str, purpose: str = "") -> str:
    return hmac.new(settings.secret_key.encode(), f"{purpose}:{value}".encode(), hashlib.sha256).hexdigest()


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def numeric_code(digits: int = 6) -> str:
    return "".join(secrets.choice("0123456789") for _ in range(digits))


def _fernet() -> Fernet:
    return Fernet(settings.get_fernet_key())


def seal(data: bytes) -> bytes:
    return _fernet().encrypt(data)


def unseal(blob: bytes) -> bytes:
    return _fernet().decrypt(blob)
