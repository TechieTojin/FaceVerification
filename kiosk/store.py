"""Kiosk local storage.

`kiosk.json` — identity: device id, server URL, key-protection mode, the protected data key,
               and the device private key encrypted with the data key.
`kiosk.db`   — SQLite: meta, subjects (templates AES-GCM encrypted with the data key), and the
               audit queue. Each audit event carries `chain_hash = SHA256(prev_hash + canonical
               event)` (verified by the server) and a local HMAC with the data key, so local
               tampering or corruption is detected before anything is used or uploaded.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from kiosk import keystore


class LocalDataError(RuntimeError):
    """Local data failed an integrity check (corruption or tampering)."""


def b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def chain_hash(prev_hash: str, ev: dict[str, Any]) -> str:
    """Must match backend.services.device_service._chain (the server re-computes it)."""
    fields = {k: ev.get(k) for k in ("event_id", "seq", "type", "occurred_at", "subject_user_id", "outcome",
                                     "reason_code", "payload", "bundle_id")}
    return hashlib.sha256((prev_hash or "").encode() + json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class Identity:
    device_id: str
    server_url: str
    protection: str
    data_key: bytes
    private_key: ec.EllipticCurvePrivateKey
    name: str = ""
    organization: str = ""


class KioskStore:
    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.identity_path = self.dir / "kiosk.json"
        self.db_path = self.dir / "kiosk.db"
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS subjects (user_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
                bundle_id TEXT NOT NULL, model TEXT NOT NULL, dim INTEGER NOT NULL, nonce BLOB NOT NULL,
                template BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, event_id TEXT UNIQUE NOT NULL,
                body TEXT NOT NULL, prev_hash TEXT NOT NULL, chain_hash TEXT NOT NULL, mac TEXT NOT NULL,
                synced_at TEXT);
        """)
        self.identity: Optional[Identity] = None

    # ── identity ────────────────────────────────────────────────────────────
    def has_identity(self) -> bool:
        return self.identity_path.exists()

    def create_identity(self, protection: str) -> tuple[ec.EllipticCurvePrivateKey, bytes]:
        data_key = os.urandom(32)
        key = ec.generate_private_key(ec.SECP256R1())
        return key, data_key

    def save_identity(self, ident: Identity) -> None:
        pem = ident.private_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                              serialization.NoEncryption())
        nonce = os.urandom(12)
        doc = {"version": 1, "device_id": ident.device_id, "server_url": ident.server_url, "name": ident.name,
               "organization": ident.organization, "protection": ident.protection,
               "data_key": b64(keystore.protect(ident.protection, ident.data_key)),
               "private_key": b64(nonce + AESGCM(ident.data_key).encrypt(nonce, pem, ident.device_id.encode()))}
        tmp = self.identity_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        tmp.replace(self.identity_path)
        self.identity = ident

    def load_identity(self) -> Identity:
        try:
            doc = json.loads(self.identity_path.read_text(encoding="utf-8"))
            data_key = keystore.unprotect(doc["protection"], unb64(doc["data_key"]))
            blob = unb64(doc["private_key"])
            pem = AESGCM(data_key).decrypt(blob[:12], blob[12:], doc["device_id"].encode())
        except keystore.KeyProtectionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LocalDataError("Kiosk identity file is corrupted") from exc
        self.identity = Identity(device_id=doc["device_id"], server_url=doc["server_url"],
                                 protection=doc["protection"], data_key=data_key,
                                 private_key=serialization.load_pem_private_key(pem, password=None),
                                 name=doc.get("name", ""), organization=doc.get("organization", ""))
        return self.identity

    # ── meta ────────────────────────────────────────────────────────────────
    def get(self, k: str, default: Any = None) -> Any:
        row = self._db.execute("SELECT v FROM meta WHERE k = ?", (k,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, k: str, v: Any) -> None:
        with self._lock:
            self._db.execute("INSERT INTO meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                             (k, json.dumps(v, default=str)))

    # ── subjects ────────────────────────────────────────────────────────────
    def replace_subjects(self, bundle: dict[str, Any]) -> None:
        key = AESGCM(self.identity.data_key)
        with self._lock:
            self._db.execute("BEGIN")
            try:
                self._db.execute("DELETE FROM subjects")
                for s in bundle["subjects"]:
                    nonce = os.urandom(12)
                    ct = key.encrypt(nonce, unb64(s["template"]), f"{s['user_id']}|{bundle['bundle_id']}".encode())
                    self._db.execute("INSERT INTO subjects VALUES (?,?,?,?,?,?,?)",
                                     (s["user_id"], s["display_name"], bundle["bundle_id"], s["model"], int(s["dim"]),
                                      nonce, ct))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        self.set("bundle", {"bundle_id": bundle["bundle_id"], "valid_until": bundle["valid_until"],
                            "issued_at": bundle["issued_at"], "subject_count": len(bundle["subjects"])})
        self.set("policy", bundle["policy"])

    def wipe_subjects(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM subjects")
            self._db.execute("DELETE FROM meta WHERE k IN ('bundle', 'policy')")

    def subjects(self) -> list[dict[str, str]]:
        return [{"user_id": u, "display_name": n}
                for u, n in self._db.execute("SELECT user_id, display_name FROM subjects ORDER BY display_name")]

    def template(self, user_id: str) -> tuple[np.ndarray, str]:
        row = self._db.execute("SELECT bundle_id, model, dim, nonce, template FROM subjects WHERE user_id = ?",
                               (user_id,)).fetchone()
        if row is None:
            raise KeyError(user_id)
        bundle_id, model, dim, nonce, ct = row
        try:
            raw = AESGCM(self.identity.data_key).decrypt(nonce, ct, f"{user_id}|{bundle_id}".encode())
        except Exception as exc:  # noqa: BLE001
            raise LocalDataError("A provisioned template failed its integrity check") from exc
        vec = np.frombuffer(raw, dtype=np.float32)
        if vec.shape[0] != dim:
            raise LocalDataError("A provisioned template has the wrong dimension")
        return vec, model

    # ── audit queue ─────────────────────────────────────────────────────────
    def _mac(self, seq: int, body: str, prev: str, chain: str) -> str:
        return hmac.new(self.identity.data_key, f"{seq}|{body}|{prev}|{chain}".encode(), hashlib.sha256).hexdigest()

    def append_event(self, type_: str, subject_user_id: Optional[str], outcome: Optional[str],
                     reason_code: Optional[str], payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            last = self._db.execute("SELECT seq, chain_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            seq, prev = (last[0] + 1, last[1]) if last else (1, "")
            bundle = self.get("bundle") or {}
            ev = {"event_id": str(uuid.uuid4()), "seq": seq, "type": type_,
                  "occurred_at": datetime.utcnow().isoformat(timespec="seconds"),
                  "subject_user_id": subject_user_id, "outcome": outcome, "reason_code": reason_code,
                  "payload": payload, "bundle_id": bundle.get("bundle_id")}
            h = chain_hash(prev, ev)
            body = json.dumps(ev, sort_keys=True, default=str)
            self._db.execute("INSERT INTO events (seq, event_id, body, prev_hash, chain_hash, mac) VALUES (?,?,?,?,?,?)",
                             (seq, ev["event_id"], body, prev, h, self._mac(seq, body, prev, h)))
            return {**ev, "prev_hash": prev, "chain_hash": h}

    def verify_events(self) -> None:
        prev = ""
        for seq, body, p, h, mac in self._db.execute("SELECT seq, body, prev_hash, chain_hash, mac FROM events ORDER BY seq"):
            if not hmac.compare_digest(mac, self._mac(seq, body, p, h)) or p != prev or chain_hash(p, json.loads(body)) != h:
                raise LocalDataError(f"Local audit log failed its integrity check at event {seq}")
            prev = h

    def pending_events(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._db.execute("SELECT body, prev_hash, chain_hash FROM events WHERE synced_at IS NULL "
                                "ORDER BY seq LIMIT ?", (limit,)).fetchall()
        return [{**json.loads(b), "prev_hash": p, "chain_hash": h} for b, p, h in rows]

    def mark_synced(self, event_ids: list[str]) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds")
        with self._lock:
            self._db.executemany("UPDATE events SET synced_at = ? WHERE event_id = ?", [(now, e) for e in event_ids])

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._db.execute("SELECT body, synced_at FROM events ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
        return [{**json.loads(b), "synced": s is not None, "synced_at": s} for b, s in rows]

    def counts(self) -> dict[str, int]:
        total, pending = self._db.execute(
            "SELECT COUNT(*), SUM(CASE WHEN synced_at IS NULL THEN 1 ELSE 0 END) FROM events").fetchone()
        return {"events": total or 0, "pending": pending or 0}

    def close(self) -> None:
        self._db.close()
