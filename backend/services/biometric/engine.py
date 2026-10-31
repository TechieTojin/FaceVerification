"""Capture-session decision engine (pure logic, no I/O).

Challenge structure (server-generated, randomised with `secrets.SystemRandom`)::

    LOOK_STRAIGHT → k random actions from {TURN_LEFT, TURN_RIGHT, BLINK} → LOOK_STRAIGHT

* LOOK_STRAIGHT is satisfied by two consecutive frontal frames that pass the
  quality gate. Passive anti-spoofing runs on the satisfying frame. For
  VERIFY/DOCUMENT an embedding is taken at the opening and closing steps; for
  ENROL, samples are taken at the opening step and the closing step collects
  the rest.
* TURN_x requires |yaw| beyond the policy angle in the requested direction
  *and* movement of at least 75 % of that angle from the yaw at step start.
  Turning the wrong way fails the round (defeats generic pre-recorded motion).
* BLINK requires an eyes-open frame followed by a frame where both eyes reach
  the closure threshold.
* Large jumps in face position/size between frames, or a second face, restart
  the round at step 0 (anti face-swap); no attempt is consumed.
* The identity decision is made only at the closing step, i.e. after liveness
  evidence is complete. With `require_bracket_match`, the opening embedding must
  also match, so the same person must be present throughout.
* A failed round (timeout, wrong action, passive spoof, mismatch with retry
  enabled) consumes one attempt and issues a *new* random challenge.

Cosine distance is a dissimilarity score, not a probability of identity.
"""
from __future__ import annotations

import json
import math
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol

import numpy as np

from backend.models.verification import OutcomeEnum
from backend.services.biometric.analysis import (
    GUIDANCE, FaceObservation, FrameAnalysis, PassiveResult, QualityReport, assess_quality,
)
from backend.services.biometric.templates import TemplateError
from backend.services.policy_service import Policy

LOOK_STRAIGHT = "LOOK_STRAIGHT"
TURN_LEFT = "TURN_LEFT"
TURN_RIGHT = "TURN_RIGHT"
BLINK = "BLINK"
ACTIVE_ACTIONS = (TURN_LEFT, TURN_RIGHT, BLINK)

INSTRUCTIONS = {
    LOOK_STRAIGHT: "Look straight at the camera",
    TURN_LEFT: "Slowly turn your head to your left",
    TURN_RIGHT: "Slowly turn your head to your right",
    BLINK: "Blink slowly",
}

LIVENESS_REASONS = {"PASSIVE_SPOOF", "WRONG_ACTION", "CHALLENGE_TIMEOUT"}
STRAIGHT_HOLD_FRAMES = 2
TRACKING_JUMP = 0.6      # face widths per frame
TRACKING_SCALE = 1.6     # size ratio between consecutive frames
TRACKING_WINDOW_S = 2.0


def generate_challenge(policy: Policy, rng: Optional[secrets.SystemRandom] = None) -> list[str]:
    if not policy.active_liveness_required:
        return [LOOK_STRAIGHT]
    rng = rng or secrets.SystemRandom()
    middle: list[str] = []
    prev = None
    for _ in range(int(policy.challenge_steps)):
        s = rng.choice([a for a in ACTIVE_ACTIONS if a != prev])
        middle.append(s)
        prev = s
    return [LOOK_STRAIGHT, *middle, LOOK_STRAIGHT]


class EngineOps(Protocol):
    def passive(self, face: FaceObservation) -> PassiveResult: ...
    def embed(self, face: FaceObservation) -> np.ndarray: ...
    def distance(self, vec: np.ndarray) -> float: ...


@dataclass
class EngineState:
    steps: list[str]
    index: int = 0
    round: int = 0
    attempts_used: int = 0
    step_started_at: Optional[str] = None
    hold: int = 0
    baseline_yaw: Optional[float] = None
    blink_open_seen: bool = False
    step_peak: float = 0.0
    last_center: Optional[list[float]] = None
    last_face_w: Optional[float] = None
    last_frame_at: Optional[str] = None
    passive_results: list[dict[str, Any]] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    quality_scores: list[float] = field(default_factory=list)
    # Not serialised to JSON — persisted encrypted by the caller.
    samples: list[np.ndarray] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("samples")
        return json.dumps(d)

    @classmethod
    def from_json(cls, raw: str, samples: Optional[list[np.ndarray]] = None) -> "EngineState":
        d = json.loads(raw)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        d = {k: v for k, v in d.items() if k in known}
        d["samples"] = list(samples or [])
        return cls(**d)

    @property
    def step(self) -> str:
        return self.steps[self.index]


@dataclass
class StepEvent:
    round: int
    index: int
    step_type: str
    status: str                     # STARTED | PASSED | FAILED
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameDecision:
    outcome: OutcomeEnum
    terminal: bool = False
    reason: Optional[str] = None
    message: str = ""
    guidance: list[str] = field(default_factory=list)
    quality: Optional[QualityReport] = None
    face: Optional[FaceObservation] = None
    passive: Optional[PassiveResult] = None
    distance: Optional[float] = None
    step_events: list[StepEvent] = field(default_factory=list)
    new_round: bool = False


def _parse(ts: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(ts) if ts else None


class Engine:
    def __init__(self, policy: Policy, purpose: str, rng: Optional[secrets.SystemRandom] = None) -> None:
        self.p = policy
        self.purpose = purpose
        self.rng = rng or secrets.SystemRandom()

    # ── public ─────────────────────────────────────────────────────────────
    def start(self, now: datetime) -> tuple[EngineState, list[StepEvent]]:
        st = EngineState(steps=generate_challenge(self.p, self.rng), step_started_at=now.isoformat())
        return st, [StepEvent(st.round, 0, st.steps[0], "STARTED")]

    def process(self, st: EngineState, analysis: FrameAnalysis, now: datetime, ops: EngineOps) -> FrameDecision:
        events: list[StepEvent] = []

        timed_out = self._step_timed_out(st, now)
        if timed_out:
            return self._round_failure(st, now, "CHALLENGE_TIMEOUT",
                                       f"Time limit for '{INSTRUCTIONS[st.step]}' exceeded", events)

        faces = analysis.faces
        if not faces:
            st.hold = 0
            return FrameDecision(OutcomeEnum.NO_FACE, reason="NO_FACE", message="No face detected",
                                 guidance=[GUIDANCE["NO_FACE"]])
        if len(faces) > 1:
            restarted = self._restart_round_progress(st, now, events)
            return FrameDecision(OutcomeEnum.MULTIPLE_FACES, reason="MULTIPLE_FACES",
                                 message="Multiple faces detected" + (" — challenge restarted" if restarted else ""),
                                 guidance=[GUIDANCE["MULTIPLE_FACES"]], step_events=events)

        face = faces[0]
        if self._tracking_lost(st, face, now):
            restarted = self._restart_round_progress(st, now, events)
            self._track(st, face, now)
            if restarted:
                return FrameDecision(OutcomeEnum.IN_PROGRESS, reason="TRACKING_LOST", face=face,
                                     message="Face moved abruptly — challenge restarted",
                                     guidance=[GUIDANCE["BLURRY"]], step_events=events)
        self._track(st, face, now)

        step = st.step
        if step == LOOK_STRAIGHT:
            return self._straight(st, face, now, ops, events)
        if step in (TURN_LEFT, TURN_RIGHT):
            return self._turn(st, face, now, events)
        return self._blink(st, face, now, events)

    def reference_needed(self) -> bool:
        return self.purpose in ("VERIFY", "DOCUMENT")

    # ── steps ──────────────────────────────────────────────────────────────
    def _straight(self, st: EngineState, face: FaceObservation, now: datetime, ops: EngineOps,
                  events: list[StepEvent]) -> FrameDecision:
        enrol = self.purpose == "ENROL"
        min_q = self.p.enrol_min_quality if enrol else self.p.verify_min_quality
        q = assess_quality(face, self.p.min_face_width_ratio, self.p.max_face_width_ratio,
                           require_frontal=True, min_score=min_q)
        if q.issues:
            st.hold = 0
            return FrameDecision(OutcomeEnum.QUALITY_FAILED, reason=q.issues[0], quality=q, face=face,
                                 message="Frame quality too low", guidance=q.guidance)
        st.hold += 1
        if st.hold < STRAIGHT_HOLD_FRAMES:
            return FrameDecision(OutcomeEnum.IN_PROGRESS, quality=q, face=face,
                                 message="Hold still", guidance=["Hold still"])

        is_opening = st.index == 0
        is_closing = st.index == len(st.steps) - 1
        passive = None
        if self.p.passive_liveness_required:
            passive = ops.passive(face)
            st.passive_results.append({"label": passive.label, "confidence": round(passive.confidence, 4)})
            if passive.label != "REAL":
                d = self._round_failure(st, now, "PASSIVE_SPOOF", "Passive anti-spoofing did not accept this frame", events)
                d.passive, d.face, d.quality = passive, face, q
                return d
        st.quality_scores.append(q.score)

        if enrol:
            st.samples.append(ops.embed(face))
            needed = int(self.p.enrol_samples)
            if is_closing and len(st.samples) >= needed:
                events.append(StepEvent(st.round, st.index, LOOK_STRAIGHT, "PASSED",
                                        {"samples": len(st.samples), "quality": round(q.score, 3)}))
                return FrameDecision(OutcomeEnum.ENROLMENT_CAPTURED, terminal=True, reason="SAMPLES_CAPTURED",
                                     quality=q, face=face, passive=passive, step_events=events,
                                     message=f"Captured {len(st.samples)} samples")
            if is_closing:
                return FrameDecision(OutcomeEnum.IN_PROGRESS, quality=q, face=face, passive=passive,
                                     message=f"Captured sample {len(st.samples)} of {needed}",
                                     guidance=["Hold still"])
            return self._advance(st, face, now, events, {"quality": round(q.score, 3)}, q, passive)

        if not is_closing and not self.p.require_bracket_match:
            return self._advance(st, face, now, events, {"quality": round(q.score, 3)}, q, passive)

        try:
            dist = ops.distance(ops.embed(face))
        except TemplateError as exc:
            return FrameDecision(OutcomeEnum.ERROR, terminal=True, reason="TEMPLATE_UNUSABLE",
                                 message=str(exc), face=face, quality=q, passive=passive)
        st.distances.append(round(dist, 6))
        if not is_closing:
            return self._advance(st, face, now, events, {"quality": round(q.score, 3)}, q, passive)

        considered = st.distances if self.p.require_bracket_match else [dist]
        final = max(considered)
        events.append(StepEvent(st.round, st.index, LOOK_STRAIGHT, "PASSED", {"quality": round(q.score, 3)}))
        if final <= self.p.match_threshold:
            return FrameDecision(OutcomeEnum.VERIFIED, terminal=True, reason="MATCH", distance=final,
                                 quality=q, face=face, passive=passive, step_events=events,
                                 message="Identity verified")
        if self.p.retry_on_mismatch and st.attempts_used + 1 < self.p.max_attempts:
            d = self._round_failure(st, now, "NOT_MATCHED", "Face did not match the reference", events,
                                    record_step=False)
            d.distance, d.face, d.quality, d.passive = final, face, q, passive
            return d
        st.attempts_used += 1
        return FrameDecision(OutcomeEnum.NOT_MATCHED, terminal=True, reason="NOT_MATCHED", distance=final,
                             quality=q, face=face, passive=passive, step_events=events,
                             message="Face did not match the reference")

    def _turn(self, st: EngineState, face: FaceObservation, now: datetime, events: list[StepEvent]) -> FrameDecision:
        size_issue = self._size_issue(face)
        if size_issue:
            return size_issue
        sign = 1.0 if st.step == TURN_LEFT else -1.0
        thr = float(self.p.yaw_threshold_degrees)
        if st.baseline_yaw is None:
            st.baseline_yaw = face.yaw
        directed = face.yaw * sign
        st.step_peak = max(st.step_peak, directed)
        if -directed >= thr:
            return self._round_failure(st, now, "WRONG_ACTION", "Head turned in the wrong direction", events,
                                       evidence={"yaw": round(face.yaw, 1)})
        moved = (face.yaw - st.baseline_yaw) * sign
        if directed >= thr and moved >= 0.75 * thr:
            return self._advance(st, face, now, events,
                                 {"yaw": round(face.yaw, 1), "baseline_yaw": round(st.baseline_yaw, 1)})
        return FrameDecision(OutcomeEnum.IN_PROGRESS, face=face, message=INSTRUCTIONS[st.step],
                             guidance=[INSTRUCTIONS[st.step]])

    def _blink(self, st: EngineState, face: FaceObservation, now: datetime, events: list[StepEvent]) -> FrameDecision:
        size_issue = self._size_issue(face)
        if size_issue:
            return size_issue
        if abs(face.yaw) > 25:
            return FrameDecision(OutcomeEnum.IN_PROGRESS, face=face, message="Face the camera to blink",
                                 guidance=[GUIDANCE["POSE"]])
        if face.closure_any < 0.35:
            st.blink_open_seen = True
        st.step_peak = max(st.step_peak, face.closure_both)
        if st.blink_open_seen and face.closure_both >= float(self.p.blink_threshold):
            return self._advance(st, face, now, events, {"closure": round(face.closure_both, 3)})
        return FrameDecision(OutcomeEnum.IN_PROGRESS, face=face, message=INSTRUCTIONS[BLINK],
                             guidance=[INSTRUCTIONS[BLINK]])

    # ── helpers ────────────────────────────────────────────────────────────
    def _size_issue(self, face: FaceObservation) -> Optional[FrameDecision]:
        q = assess_quality(face, self.p.min_face_width_ratio, self.p.max_face_width_ratio, require_frontal=False)
        blocking = [i for i in q.issues if i in ("FACE_TOO_SMALL", "FACE_TOO_LARGE", "OUT_OF_FRAME", "TOO_DARK")]
        if blocking:
            return FrameDecision(OutcomeEnum.QUALITY_FAILED, reason=blocking[0], quality=q, face=face,
                                 message="Adjust position", guidance=[GUIDANCE[b] for b in blocking])
        return None

    def _advance(self, st: EngineState, face: FaceObservation, now: datetime, events: list[StepEvent],
                 evidence: dict[str, Any], q: Optional[QualityReport] = None,
                 passive: Optional[PassiveResult] = None) -> FrameDecision:
        events.append(StepEvent(st.round, st.index, st.step, "PASSED", evidence))
        passed_label = st.step.replace("_", " ").capitalize()
        st.index += 1
        st.hold = 0
        st.step_started_at = now.isoformat()
        st.baseline_yaw = face.yaw
        st.blink_open_seen = False
        st.step_peak = 0.0
        events.append(StepEvent(st.round, st.index, st.step, "STARTED"))
        return FrameDecision(OutcomeEnum.IN_PROGRESS, reason="STEP_PASSED", face=face, quality=q, passive=passive,
                             message=f"{passed_label} passed", guidance=[INSTRUCTIONS[st.step]], step_events=events)

    def _round_failure(self, st: EngineState, now: datetime, reason: str, message: str,
                       events: list[StepEvent], evidence: Optional[dict] = None,
                       record_step: bool = True) -> FrameDecision:
        if record_step:
            events.append(StepEvent(st.round, st.index, st.step, "FAILED", {"reason": reason, **(evidence or {})}))
        st.attempts_used += 1
        if st.attempts_used >= int(self.p.max_attempts):
            outcome = OutcomeEnum.LIVENESS_FAILED if reason in LIVENESS_REASONS else OutcomeEnum.NOT_MATCHED
            return FrameDecision(outcome, terminal=True, reason=reason, message=message, step_events=events)
        self._new_round(st, now)
        events.append(StepEvent(st.round, 0, st.steps[0], "STARTED"))
        remaining = int(self.p.max_attempts) - st.attempts_used
        return FrameDecision(OutcomeEnum.RETRY_REQUIRED, reason=reason, new_round=True, step_events=events,
                             message=f"{message}. New challenge issued ({remaining} attempt(s) left)",
                             guidance=[INSTRUCTIONS[st.steps[0]]])

    def _new_round(self, st: EngineState, now: datetime) -> None:
        st.round += 1
        st.steps = generate_challenge(self.p, self.rng)
        st.index = 0
        st.hold = 0
        st.baseline_yaw = None
        st.blink_open_seen = False
        st.step_peak = 0.0
        st.step_started_at = now.isoformat()
        st.distances = []
        st.passive_results = []
        st.quality_scores = []
        st.samples = []

    def _restart_round_progress(self, st: EngineState, now: datetime, events: list[StepEvent]) -> bool:
        """Reset to step 0 of the current round without consuming an attempt."""
        if st.index == 0 and not st.samples and not st.distances:
            st.hold = 0
            return False
        events.append(StepEvent(st.round, st.index, st.step, "FAILED", {"reason": "TRACKING_RESET"}))
        st.index = 0
        st.hold = 0
        st.baseline_yaw = None
        st.blink_open_seen = False
        st.step_peak = 0.0
        st.step_started_at = now.isoformat()
        st.distances = []
        st.passive_results = []
        st.quality_scores = []
        st.samples = []
        events.append(StepEvent(st.round, 0, st.steps[0], "STARTED"))
        return True

    def _step_timed_out(self, st: EngineState, now: datetime) -> bool:
        if st.index == 0:
            return False  # positioning for the opening step is bounded by session/inactivity timeouts
        started = _parse(st.step_started_at)
        if started is None:
            return False
        limit = float(self.p.challenge_step_timeout_seconds)
        if self.purpose == "ENROL" and st.index == len(st.steps) - 1:
            limit *= 2
        return (now - started).total_seconds() > limit

    def _tracking_lost(self, st: EngineState, face: FaceObservation, now: datetime) -> bool:
        last_at = _parse(st.last_frame_at)
        if st.last_center is None or last_at is None or st.last_face_w is None:
            return False
        if (now - last_at).total_seconds() > TRACKING_WINDOW_S:
            return False
        cx, cy = face.center
        w = max(face.bbox[2], st.last_face_w, 1e-6)
        jump = math.hypot(cx - st.last_center[0], cy - st.last_center[1]) / w
        scale = max(face.bbox[2], st.last_face_w) / max(min(face.bbox[2], st.last_face_w), 1e-6)
        return jump > TRACKING_JUMP or scale > TRACKING_SCALE

    @staticmethod
    def _track(st: EngineState, face: FaceObservation, now: datetime) -> None:
        st.last_center = list(face.center)
        st.last_face_w = face.bbox[2]
        st.last_frame_at = now.isoformat()


def step_view(st: EngineState) -> dict[str, Any]:
    return {
        "round": st.round,
        "index": st.index,
        "total": len(st.steps),
        "type": st.step,
        "instruction": INSTRUCTIONS[st.step],
        "steps": st.steps,
        "progress": round(st.index / max(1, len(st.steps)), 3),
    }
