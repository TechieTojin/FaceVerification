"""Local key protection for the kiosk's data key.

* `dpapi`      — Windows Data Protection API, user scope (OS-protected; not hardware-backed).
* `passphrase` — scrypt-derived key from KIOSK_PASSPHRASE (for non-Windows hosts and tests).
The data key (32 random bytes) encrypts the device private key, provisioned templates and
authenticates the local audit log.
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


class KeyProtectionError(RuntimeError):
    pass


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _dpapi(data: bytes, protect: bool) -> bytes:
    if sys.platform != "win32":
        raise KeyProtectionError("DPAPI is only available on Windows")
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    buf = ctypes.create_string_buffer(data, len(data))
    blob_in = _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    blob_out = _Blob()
    entropy = b"FaceVerification-kiosk-v1"
    ebuf = ctypes.create_string_buffer(entropy, len(entropy))
    blob_ent = _Blob(len(entropy), ctypes.cast(ebuf, ctypes.POINTER(ctypes.c_char)))
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    ok = fn(ctypes.byref(blob_in), None, ctypes.byref(blob_ent), None, None, 0x1, ctypes.byref(blob_out))
    if not ok:
        raise KeyProtectionError(f"DPAPI {'protect' if protect else 'unprotect'} failed")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


def _passphrase_key(salt: bytes) -> bytes:
    phrase = os.environ.get("KIOSK_PASSPHRASE", "")
    if len(phrase) < 12:
        raise KeyProtectionError("KIOSK_PASSPHRASE must be set (at least 12 characters) for passphrase protection")
    return Scrypt(salt=salt, length=32, n=2 ** 15, r=8, p=1).derive(phrase.encode())


def protect(mode: str, data: bytes) -> bytes:
    if mode == "dpapi":
        return _dpapi(data, True)
    if mode == "passphrase":
        salt, nonce = os.urandom(16), os.urandom(12)
        return salt + nonce + AESGCM(_passphrase_key(salt)).encrypt(nonce, data, b"kiosk-data-key")
    raise KeyProtectionError(f"Unknown key protection mode {mode}")


def unprotect(mode: str, blob: bytes) -> bytes:
    try:
        if mode == "dpapi":
            return _dpapi(blob, False)
        if mode == "passphrase":
            salt, nonce, ct = blob[:16], blob[16:28], blob[28:]
            return AESGCM(_passphrase_key(salt)).decrypt(nonce, ct, b"kiosk-data-key")
    except KeyProtectionError:
        raise
    except Exception as exc:  # noqa: BLE001 — wrong passphrase, other user, corrupted blob
        raise KeyProtectionError("The kiosk data key could not be unlocked") from exc
    raise KeyProtectionError(f"Unknown key protection mode {mode}")


def default_mode() -> str:
    return "dpapi" if sys.platform == "win32" else "passphrase"


DESCRIPTIONS = {
    "dpapi": "Windows DPAPI (user scope) — OS-protected, not hardware-backed",
    "passphrase": "Passphrase-derived key (scrypt) — software protection, not hardware-backed",
}
