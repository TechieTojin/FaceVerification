"""Kiosk agent: local verification sessions, offline status, synchronisation.

Must be imported after `kiosk.runtime_env.prepare()` (it imports backend modules).
"""
from __future__ import annotations

import hashlib
import logging
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np

from backend.models.verification import OutcomeEnum
from backend.services.biometric.analysis import FaceObservation, PassiveResult
from backend.services.biometric.engine import Engine, EngineState, step_view
from backend.services.biometric.imaging import InvalidImageError, decode_image
from backend.services.biometric.templates import cosine_distance, l2_normalise
from backend.services.inference import build_provider
from backend.services.policy_service import Policy
from kiosk import VERSION, client, keystore
from kiosk.store import KioskStore, LocalDataError

log = logging.getLogger("kiosk")

CLOCK_ROLLBACK_TOLERANCE = timedelta(minutes=5)
MAX_SESSION_FRAMES_REPLAY = 5


class KioskError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass
class LocalSession:
    id: str
    user_id: str
    display_name: str
    engine: Engine
    state: EngineState
    policy: Policy
    template: np.ndarray
    created_at: datetime
    expires_at: datetime
    token_hash: str
    seq: int = 0
    frames: int = 0
    active: bool = True
    outcome: Optional[str] = None
    reason: Optional[str] = None
    digests: set = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)


class _Ops:
    def __init__(self, provider, bgr, template):
        self.provider, self.bgr, self.template = provider, bgr, template

    def passive(self, face: FaceObservation) -> PassiveResult:
        return self.provider.passive_liveness(self.bgr, face)

    def embed(self, face: FaceObservation) -> np.ndarray:
        return self.provider.embed(self.bgr, face)

    def distance(self, vec: np.ndarray) -> float:
        return cosine_distance(l2_normalise(vec), self.template)


class KioskAgent:
    def __init__(self, store: KioskStore):
        self.store = store
        self.ident = store.load_identity()
        self.store.verify_events()
        self.provider = build_provider()
        self.provider.warmup()
        self.sessions: dict[str, LocalSession] = {}
        self.online = False
        self.revoked = bool(store.get("revoked", False))
        self.last_sync_error: Optional[str] = None
        self._sync_lock = threading.Lock()

    # ── status / validity ───────────────────────────────────────────────────
    def _now(self) -> datetime:
        now = datetime.utcnow()
        last = self.store.get("last_seen_time")
        if last and now < datetime.fromisoformat(last) - CLOCK_ROLLBACK_TOLERANCE:
            raise KioskError(423, "CLOCK_ROLLBACK", "The kiosk clock moved backwards; reconnect to the server")
        if not last or now > datetime.fromisoformat(last):
            self.store.set("last_seen_time", now.isoformat())
        return now

    def bundle_state(self) -> dict[str, Any]:
        b = self.store.get("bundle")
        if not b:
            return {"provisioned": False, "valid": False, "reason": "NO_BUNDLE"}
        valid_until = datetime.fromisoformat(b["valid_until"].replace("Z", ""))
        expired = datetime.utcnow() > valid_until
        return {"provisioned": True, "valid": not expired, "reason": "EXPIRED" if expired else None,
                "bundle_id": b["bundle_id"], "valid_until": b["valid_until"], "issued_at": b["issued_at"],
                "subject_count": b["subject_count"]}

    def status(self) -> dict[str, Any]:
        return {"device_id": self.ident.device_id, "name": self.ident.name, "organization": self.ident.organization,
                "server_url": self.ident.server_url, "online": self.online, "revoked": self.revoked,
                "mode": "OFFLINE" if not self.online else "ONLINE", "bundle": self.bundle_state(),
                "events": self.store.counts(), "last_sync_at": self.store.get("last_sync_at"),
                "last_sync_error": self.last_sync_error, "software_version": VERSION,
                "key_protection": keystore.DESCRIPTIONS.get(self.ident.protection, self.ident.protection),
                "hardware_backed": False, "inference": getattr(self.provider, "name", "deepface"),
                "inference_location": "on this device"}

    # ── local verification ──────────────────────────────────────────────────
    def start_session(self, user_id: str) -> dict[str, Any]:
        if self.revoked:
            raise KioskError(403, "DEVICE_REVOKED", "This kiosk has been revoked")
        now = self._now()
        b = self.bundle_state()
        if not b["valid"]:
            self.store.append_event("VERIFICATION_REFUSED", user_id, None, b["reason"], {"offline": not self.online})
            raise KioskError(423, "OFFLINE_AUTHORIZATION_EXPIRED" if b["reason"] == "EXPIRED" else "NOT_PROVISIONED",
                             "Offline authorisation has expired. Reconnect this kiosk to the server."
                             if b["reason"] == "EXPIRED" else "This kiosk has no provisioned subjects yet.")
        try:
            template, _model = self.store.template(user_id)
        except KeyError:
            raise KioskError(404, "SUBJECT_NOT_PROVISIONED", "This person is not authorised on this kiosk") from None
        except LocalDataError as exc:
            self.store.append_event("INTEGRITY_FAILURE", user_id, None, "LOCAL_DATA_CORRUPTED", {})
            raise KioskError(500, "LOCAL_DATA_CORRUPTED", str(exc)) from exc
        policy = Policy.from_json(_policy_json(self.store.get("policy")))
        for s in list(self.sessions.values()):
            if s.active:
                self._close(s, OutcomeEnum.SESSION_CANCELLED.value, "SUPERSEDED")
        engine = Engine(policy, "VERIFY")
        state, _ = engine.start(now)
        token = secrets.token_urlsafe(24)
        name = next((x["display_name"] for x in self.store.subjects() if x["user_id"] == user_id), "")
        s = LocalSession(id=secrets.token_hex(16), user_id=user_id, display_name=name, engine=engine, state=state,
                         policy=policy, template=l2_normalise(template.astype(np.float64)), created_at=now,
                         expires_at=now + timedelta(seconds=int(policy.session_timeout_seconds)),
                         token_hash=hashlib.sha256(token.encode()).hexdigest())
        self.sessions[s.id] = s
        return {**self._view(s), "frame_token": token}

    def submit_frame(self, session_id: str, seq: int, token: str, data: bytes) -> dict[str, Any]:
        s = self.sessions.get(session_id)
        if s is None:
            raise KioskError(404, "SESSION_NOT_FOUND", "Session not found")
        if not s.lock.acquire(blocking=False):
            raise KioskError(409, "FRAME_IN_FLIGHT", "A frame for this session is already being processed")
        try:
            now = self._now()
            if not s.active:
                return self._view(s)
            if now > s.expires_at:
                self._close(s, OutcomeEnum.SESSION_EXPIRED.value, "SESSION_TIMEOUT")
                return self._view(s)
            if seq != s.seq + 1 or hashlib.sha256(token.encode()).hexdigest() != s.token_hash:
                raise KioskError(409, "STALE_FRAME", "Frame token or sequence is not current")
            if s.frames >= int(s.policy.max_frames_per_session):
                self._close(s, OutcomeEnum.ATTEMPTS_EXHAUSTED.value, "FRAME_LIMIT")
                return self._view(s)
            digest = hashlib.sha256(data).hexdigest()
            if digest in s.digests:
                raise KioskError(409, "FRAME_REPLAYED", "The camera image is identical to an earlier frame")
            s.digests.add(digest)
            try:
                bgr = decode_image(data, max_side=1280)
            except InvalidImageError as exc:
                raise KioskError(422, "INVALID_IMAGE", str(exc)) from exc
            s.seq, s.frames = seq, s.frames + 1
            analysis = self.provider.analyze(bgr)
            decision = s.engine.process(s.state, analysis, now, _Ops(self.provider, bgr, s.template))
            new_token = secrets.token_urlsafe(24)
            s.token_hash = hashlib.sha256(new_token.encode()).hexdigest()
            frame = {"seq": seq, "outcome": decision.outcome.value, "reason": decision.reason,
                     "message": decision.message, "guidance": decision.guidance, "terminal": decision.terminal,
                     "new_round": decision.new_round, "face_count": len(analysis.faces),
                     "face_box": list(decision.face.bbox) if decision.face else None,
                     "pose": ({"yaw": round(decision.face.yaw, 1), "pitch": round(decision.face.pitch, 1)}
                              if decision.face else None),
                     "quality": ({"score": decision.quality.score, "issues": decision.quality.issues,
                                  "components": decision.quality.components} if decision.quality else None)}
            if decision.terminal:
                self._close(s, decision.outcome.value, decision.reason, distance=decision.distance)
            view = self._view(s, frame)
            if not decision.terminal:
                view["frame_token"] = new_token
            return view
        finally:
            s.lock.release()

    def cancel(self, session_id: str) -> dict[str, Any]:
        s = self.sessions.get(session_id)
        if s is None:
            raise KioskError(404, "SESSION_NOT_FOUND", "Session not found")
        if s.active:
            self._close(s, OutcomeEnum.SESSION_CANCELLED.value, "USER_CANCELLED")
        return self._view(s)

    def _close(self, s: LocalSession, outcome: str, reason: Optional[str], distance: Optional[float] = None) -> None:
        s.active, s.outcome, s.reason = False, outcome, reason
        self.store.append_event("VERIFICATION", s.user_id, outcome, reason, {
            "offline": not self.online, "frames": s.frames, "attempts": s.state.attempts_used,
            "steps": s.state.steps, "policy_version": s.policy.version,
            "distance": round(float(distance), 4) if distance is not None else None,
            "inference": getattr(self.provider, "name", "deepface")})

    def _view(self, s: LocalSession, frame: Optional[dict] = None) -> dict[str, Any]:
        return {"session_id": s.id, "purpose": "VERIFY", "channel": "KIOSK_OFFLINE", "target_user_id": s.user_id,
                "subject_name": s.display_name, "status": "ACTIVE" if s.active else "CLOSED",
                "final_outcome": s.outcome, "reason_code": s.reason, "created_at": s.created_at,
                "expires_at": s.expires_at, "frame_seq": s.seq, "frames_processed": s.frames,
                "frames_remaining": max(0, int(s.policy.max_frames_per_session) - s.frames),
                "attempts_used": s.state.attempts_used,
                "attempts_remaining": max(0, int(s.policy.max_attempts) - s.state.attempts_used),
                "policy_version": s.policy.version, "capture": s.policy.capture_config(),
                "challenge": step_view(s.state) if s.active else None, "frame": frame, "review_status": None,
                "camera_id": None, "document_check_id": None, "completed_at": None if s.active else datetime.utcnow()}

    # ── synchronisation ─────────────────────────────────────────────────────
    def sync(self, refresh_bundle: bool = True) -> dict[str, Any]:
        with self._sync_lock:
            out: dict[str, Any] = {"uploaded": 0, "duplicates": 0, "bundle_refreshed": False}
            try:
                client.signed(self.ident, "POST", "/api/v1/device-api/heartbeat",
                              {"software_version": VERSION, "health": {"pending_events": self.store.counts()["pending"],
                                                                      "bundle_valid": self.bundle_state()["valid"]}})
                self.online = True
                pending = self.store.pending_events()
                if pending:
                    res = client.signed(self.ident, "POST", "/api/v1/device-api/offline-events", {"events": pending})
                    done = list(res.get("acked", [])) + list(res.get("duplicates", []))
                    self.store.mark_synced(done)
                    out["uploaded"], out["duplicates"] = len(res.get("acked", [])), len(res.get("duplicates", []))
                    out["integrity_issues"] = res.get("integrity_issues", 0)
                if refresh_bundle:
                    resp = client.signed(self.ident, "POST", "/api/v1/device-api/offline-bundle")
                    jwks = client.fetch_jwks(self.ident.server_url)
                    self.store.replace_subjects(client.open_bundle(self.ident, resp, jwks))
                    out["bundle_refreshed"] = True
                self.store.set("last_sync_at", datetime.utcnow().isoformat(timespec="seconds"))
                self.last_sync_error = None
            except client.ServerUnreachable as exc:
                self.online = False
                self.last_sync_error = "Server unreachable — working offline"
                out["error"] = "SERVER_UNREACHABLE"
                log.info("Sync skipped: %s", exc)
            except client.ServerError as exc:
                self.last_sync_error = f"{exc.code}: {exc.message}"
                out["error"] = exc.code
                if exc.code == "DEVICE_REVOKED":
                    self.revoked = True
                    self.online = True
                    self.store.set("revoked", True)
                    self.store.wipe_subjects()          # templates are erased as soon as revocation is known
                    for s in self.sessions.values():
                        if s.active:
                            self._close(s, OutcomeEnum.SESSION_CANCELLED.value, "DEVICE_REVOKED")
            return out

    def recent_events(self) -> list[dict[str, Any]]:
        return self.store.recent_events()


def _policy_json(raw: Any) -> str:
    import json
    return json.dumps(raw) if not isinstance(raw, str) else raw
