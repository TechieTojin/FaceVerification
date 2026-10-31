"""A minimal software WebAuthn authenticator for tests (ES256, "none" attestation).

It produces real attestation objects and signed assertions, so the server's
py_webauthn verification (challenge, origin, RP ID hash, flags, signature,
sign counter) runs unmodified.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field

import cbor2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from webauthn.helpers import base64url_to_bytes, bytes_to_base64url

UP, UV, AT = 0x01, 0x04, 0x40


@dataclass
class SoftAuthenticator:
    rp_id: str = "localhost"
    origin: str = "http://localhost:5173"
    credential_id: bytes = field(default_factory=lambda: os.urandom(32))
    key: ec.EllipticCurvePrivateKey = field(default_factory=lambda: ec.generate_private_key(ec.SECP256R1()))
    sign_count: int = 0
    user_handle: bytes = b""

    def _cose_key(self) -> bytes:
        nums = self.key.public_key().public_numbers()
        return cbor2.dumps({1: 2, 3: -7, -1: 1, -2: nums.x.to_bytes(32, "big"), -3: nums.y.to_bytes(32, "big")})

    def _rp_hash(self) -> bytes:
        return hashlib.sha256(self.rp_id.encode()).digest()

    def create(self, options: dict, origin: str | None = None) -> dict:
        self.user_handle = base64url_to_bytes(options["user"]["id"])
        client_data = json.dumps({"type": "webauthn.create", "challenge": options["challenge"],
                                  "origin": origin or self.origin, "crossOrigin": False}).encode()
        attested = (b"\x00" * 16 + struct.pack(">H", len(self.credential_id)) + self.credential_id + self._cose_key())
        auth_data = self._rp_hash() + bytes([UP | UV | AT]) + struct.pack(">I", self.sign_count) + attested
        att_obj = cbor2.dumps({"fmt": "none", "attStmt": {}, "authData": auth_data})
        cid = bytes_to_base64url(self.credential_id)
        return {"id": cid, "rawId": cid, "type": "public-key",
                "response": {"clientDataJSON": bytes_to_base64url(client_data),
                             "attestationObject": bytes_to_base64url(att_obj), "transports": ["internal"]},
                "clientExtensionResults": {}, "authenticatorAttachment": "platform"}

    def get(self, options: dict, origin: str | None = None, flags: int = UP | UV, bump: int = 1,
            challenge: str | None = None) -> dict:
        self.sign_count += bump
        client_data = json.dumps({"type": "webauthn.get", "challenge": challenge or options["challenge"],
                                  "origin": origin or self.origin, "crossOrigin": False}).encode()
        auth_data = self._rp_hash() + bytes([flags]) + struct.pack(">I", self.sign_count)
        sig = self.key.sign(auth_data + hashlib.sha256(client_data).digest(), ec.ECDSA(hashes.SHA256()))
        cid = bytes_to_base64url(self.credential_id)
        return {"id": cid, "rawId": cid, "type": "public-key",
                "response": {"clientDataJSON": bytes_to_base64url(client_data),
                             "authenticatorData": bytes_to_base64url(auth_data),
                             "signature": bytes_to_base64url(sig),
                             "userHandle": bytes_to_base64url(self.user_handle) if self.user_handle else None},
                "clientExtensionResults": {}, "authenticatorAttachment": "platform"}
