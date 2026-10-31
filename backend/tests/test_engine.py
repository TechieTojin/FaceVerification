"""Unit tests of the decision engine with deterministic observations and embeddings.

These exercise the *real* state machine; only the model calls (embed / passive)
are replaced by deterministic functions.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pytest

from backend.models.verification import OutcomeEnum
from backend.services.biometric.analysis import FaceObservation, FrameAnalysis, PassiveResult
from backend.services.biometric.engine import (
    BLINK, LOOK_STRAIGHT, TURN_LEFT, TURN_RIGHT, Engine, generate_challenge,
)
from backend.services.biometric.providers import fake_identity_vector
from backend.services.biometric.templates import TemplateMismatchError, cosine_distance
from backend.services.policy_service import DEFAULTS, Policy

T0 = datetime(2026, 1, 1, 12, 0, 0)


def policy(**overrides) -> Policy:
    v = dict(DEFAULTS)
    v.update(overrides)
    return Policy(1, v)


def face(yaw=0.0, blink=0.0, identity=1, real=True, quality=0.9, width=0.36, x=None, pitch=0.0) -> FaceObservation:
    x = 0.5 - width / 2 if x is None else x
    return FaceObservation(bbox=(x, 0.25, width, width * 1.3), yaw=yaw, pitch=pitch, roll=0.0,
                           blink_left=blink, blink_right=blink, sharpness=quality, brightness=0.25 + 0.5 * quality,
                           extra={"identity": identity, "real": real})


def frame(*faces) -> FrameAnalysis:
    return FrameAnalysis(width=640, height=480, faces=list(faces))


class Ops:
    def __init__(self, reference_identity=1, fail_embed=False):
        self.ref = fake_identity_vector(reference_identity)
        self.fail_embed = fail_embed
        self.embeds = 0
        self.passives = 0

    def passive(self, f):
        self.passives += 1
        return PassiveResult("REAL" if f.extra["real"] else "SPOOF", 0.93)

    def embed(self, f):
        self.embeds += 1
        if self.fail_embed:
            raise TemplateMismatchError("dimension mismatch")
        return fake_identity_vector(f.extra["identity"])

    def distance(self, vec):
        return cosine_distance(vec, self.ref)


def obs_for(step, phase, identity=1, real=True):
    if step == TURN_LEFT:
        return face(yaw=25, identity=identity, real=real)
    if step == TURN_RIGHT:
        return face(yaw=-25, identity=identity, real=real)
    if step == BLINK:
        return face(blink=0.0 if phase % 2 == 0 else 1.0, identity=identity, real=real)
    return face(identity=identity, real=real)


def run(engine, st, ops, identity=1, real=True, limit=60, step_identity=None):
    now = T0
    d = None
    for i in range(limit):
        now += timedelta(milliseconds=400)
        ident = step_identity(st) if step_identity else identity
        d = engine.process(st, frame(obs_for(st.step, i, ident, real)), now, ops)
        if d.terminal:
            return d
    return d


# ── challenge generation ─────────────────────────────────────────────────

def test_challenge_brackets_and_no_consecutive_repeats():
    p = policy(challenge_steps=4)
    for seed in range(50):
        steps = generate_challenge(p, random.Random(seed))
        assert steps[0] == LOOK_STRAIGHT and steps[-1] == LOOK_STRAIGHT
        assert len(steps) == 6
        middle = steps[1:-1]
        assert all(s in (TURN_LEFT, TURN_RIGHT, BLINK) for s in middle)
        assert all(a != b for a, b in zip(middle, middle[1:]))


def test_challenge_is_randomised():
    p = policy(challenge_steps=3)
    seen = {tuple(generate_challenge(p)) for _ in range(40)}
    assert len(seen) > 3


def test_active_liveness_disabled_gives_single_step():
    assert generate_challenge(policy(active_liveness_required=False)) == [LOOK_STRAIGHT]


# ── verification decisions ───────────────────────────────────────────────

def test_genuine_face_is_verified_after_full_challenge():
    eng = Engine(policy(), "VERIFY")
    st, _ = eng.start(T0)
    ops = Ops(reference_identity=1)
    d = run(eng, st, ops, identity=1)
    assert d.outcome == OutcomeEnum.VERIFIED and d.terminal
    assert d.distance is not None and d.distance <= 0.01
    assert ops.embeds == 2           # opening + closing (bracket)
    assert ops.passives == 2


def test_impostor_is_not_matched():
    eng = Engine(policy(), "VERIFY")
    st, _ = eng.start(T0)
    d = run(eng, st, Ops(reference_identity=1), identity=2)
    assert d.outcome == OutcomeEnum.NOT_MATCHED and d.terminal
    assert d.distance > 0.68


def test_decision_made_only_after_liveness_steps_complete():
    eng = Engine(policy(challenge_steps=2), "VERIFY")
    st, _ = eng.start(T0)
    ops = Ops(1)
    now = T0
    outcomes = []
    for i in range(60):
        now += timedelta(milliseconds=400)
        d = eng.process(st, frame(obs_for(st.step, i, identity=2)), now, ops)
        outcomes.append(d.outcome)
        if d.terminal:
            break
    assert outcomes[-1] == OutcomeEnum.NOT_MATCHED
    assert OutcomeEnum.NOT_MATCHED not in outcomes[:-1]


def test_bracket_mismatch_detects_face_swap_mid_challenge():
    """Opening frame from an impostor, closing frame from the genuine user → rejected."""
    eng = Engine(policy(), "VERIFY")
    st, _ = eng.start(T0)
    d = run(eng, st, Ops(1), step_identity=lambda s: 2 if s.index == 0 else 1)
    assert d.outcome == OutcomeEnum.NOT_MATCHED


def test_bracket_disabled_uses_closing_frame_only():
    eng = Engine(policy(require_bracket_match=False), "VERIFY")
    st, _ = eng.start(T0)
    ops = Ops(1)
    d = run(eng, st, ops, step_identity=lambda s: 2 if s.index == 0 else 1)
    assert d.outcome == OutcomeEnum.VERIFIED
    assert ops.embeds == 1


def test_threshold_applies_exactly():
    # identity 15 is at cosine distance 0.55 from identity 1
    for thr, expected in ((0.68, OutcomeEnum.VERIFIED), (0.50, OutcomeEnum.NOT_MATCHED)):
        eng = Engine(policy(match_threshold=thr), "VERIFY")
        st, _ = eng.start(T0)
        d = run(eng, st, Ops(1), identity=15)
        assert d.outcome == expected, thr
        assert d.distance == pytest.approx(0.55, abs=1e-4)


def test_retry_on_mismatch_issues_new_challenge_then_terminal():
    eng = Engine(policy(retry_on_mismatch=True, max_attempts=2), "VERIFY")
    st, _ = eng.start(T0)
    seen_retry = False
    now = T0
    for i in range(80):
        now += timedelta(milliseconds=400)
        d = eng.process(st, frame(obs_for(st.step, i, identity=2)), now, Ops(1))
        if d.outcome == OutcomeEnum.RETRY_REQUIRED:
            seen_retry = True
            assert d.new_round
        if d.terminal:
            break
    assert seen_retry
    assert d.outcome == OutcomeEnum.NOT_MATCHED and st.attempts_used == 2


# ── liveness ─────────────────────────────────────────────────────────────

def test_spoof_detected_by_passive_check_exhausts_attempts():
    eng = Engine(policy(max_attempts=2), "VERIFY")
    st, _ = eng.start(T0)
    d = run(eng, st, Ops(1), real=False)
    assert d.outcome == OutcomeEnum.LIVENESS_FAILED and d.reason == "PASSIVE_SPOOF"
    assert st.attempts_used == 2


def test_passive_disabled_accepts_spoof_label():
    eng = Engine(policy(passive_liveness_required=False), "VERIFY")
    st, _ = eng.start(T0)
    ops = Ops(1)
    d = run(eng, st, ops, real=False)
    assert d.outcome == OutcomeEnum.VERIFIED and ops.passives == 0


def _engine_with_steps(steps, **pol):
    eng = Engine(policy(**pol), "VERIFY")
    st, _ = eng.start(T0)
    st.steps = steps
    return eng, st


def _advance_opening(eng, st, ops, now):
    for _ in range(2):
        now += timedelta(milliseconds=400)
        eng.process(st, frame(face()), now, ops)
    assert st.index == 1
    return now


def test_wrong_direction_fails_round():
    eng, st = _engine_with_steps([LOOK_STRAIGHT, TURN_LEFT, LOOK_STRAIGHT], max_attempts=3)
    ops = Ops(1)
    now = _advance_opening(eng, st, ops, T0)
    d = eng.process(st, frame(face(yaw=-25)), now + timedelta(milliseconds=400), ops)
    assert d.outcome == OutcomeEnum.RETRY_REQUIRED and d.reason == "WRONG_ACTION"
    assert st.attempts_used == 1 and st.index == 0 and st.round == 1


def test_turn_requires_minimum_movement_from_baseline():
    eng, st = _engine_with_steps([LOOK_STRAIGHT, TURN_LEFT, LOOK_STRAIGHT], yaw_threshold_degrees=18)
    ops = Ops(1)
    now = _advance_opening(eng, st, ops, T0)
    st.baseline_yaw = 15.0   # already turned at step start
    d = eng.process(st, frame(face(yaw=20)), now + timedelta(milliseconds=400), ops)
    assert d.outcome == OutcomeEnum.IN_PROGRESS and st.index == 1
    d = eng.process(st, frame(face(yaw=30)), now + timedelta(milliseconds=800), ops)
    assert st.index == 2


def test_blink_requires_open_then_closed():
    eng, st = _engine_with_steps([LOOK_STRAIGHT, BLINK, LOOK_STRAIGHT])
    ops = Ops(1)
    now = _advance_opening(eng, st, ops, T0)
    st.blink_open_seen = False
    eng.process(st, frame(face(blink=1.0)), now + timedelta(milliseconds=300), ops)
    assert st.index == 1, "closed eyes without a preceding open frame must not pass"
    eng.process(st, frame(face(blink=0.0)), now + timedelta(milliseconds=600), ops)
    eng.process(st, frame(face(blink=1.0)), now + timedelta(milliseconds=900), ops)
    assert st.index == 2


def test_step_timeout_fails_round():
    eng, st = _engine_with_steps([LOOK_STRAIGHT, TURN_RIGHT, LOOK_STRAIGHT], challenge_step_timeout_seconds=5, max_attempts=1)
    ops = Ops(1)
    now = _advance_opening(eng, st, ops, T0)
    d = eng.process(st, frame(face()), now + timedelta(seconds=6), ops)
    assert d.outcome == OutcomeEnum.LIVENESS_FAILED and d.reason == "CHALLENGE_TIMEOUT"


def test_multiple_faces_reset_progress():
    eng, st = _engine_with_steps([LOOK_STRAIGHT, TURN_LEFT, LOOK_STRAIGHT])
    ops = Ops(1)
    now = _advance_opening(eng, st, ops, T0)
    d = eng.process(st, frame(face(x=0.05, width=0.3), face(x=0.6, width=0.3)), now + timedelta(milliseconds=400), ops)
    assert d.outcome == OutcomeEnum.MULTIPLE_FACES
    assert st.index == 0 and st.distances == [] and st.attempts_used == 0


def test_abrupt_face_jump_restarts_round():
    eng, st = _engine_with_steps([LOOK_STRAIGHT, TURN_LEFT, LOOK_STRAIGHT])
    ops = Ops(1)
    now = _advance_opening(eng, st, ops, T0)
    d = eng.process(st, frame(face(x=0.62, width=0.36, yaw=25)), now + timedelta(milliseconds=300), ops)
    assert d.reason == "TRACKING_LOST" and st.index == 0


def test_no_face_does_not_advance():
    eng = Engine(policy(), "VERIFY")
    st, _ = eng.start(T0)
    d = eng.process(st, frame(), T0 + timedelta(seconds=1), Ops(1))
    assert d.outcome == OutcomeEnum.NO_FACE and not d.terminal and st.index == 0


def test_quality_gate_blocks_identity_frames_with_guidance():
    eng = Engine(policy(), "VERIFY")
    st, _ = eng.start(T0)
    d = eng.process(st, frame(face(width=0.12)), T0 + timedelta(seconds=1), Ops(1))
    assert d.outcome == OutcomeEnum.QUALITY_FAILED and "Move closer" in d.guidance
    d = eng.process(st, frame(face(quality=0.05)), T0 + timedelta(seconds=2), Ops(1))
    assert d.outcome == OutcomeEnum.QUALITY_FAILED
    assert any(g in d.guidance for g in ("Hold still", "Improve lighting"))
    d = eng.process(st, frame(face(yaw=30)), T0 + timedelta(seconds=3), Ops(1))
    assert "Look directly at the camera" in d.guidance


def test_template_error_is_terminal_error_not_match():
    eng = Engine(policy(active_liveness_required=False), "VERIFY")
    st, _ = eng.start(T0)
    ops = Ops(1, fail_embed=True)
    d = run(eng, st, ops)
    assert d.outcome == OutcomeEnum.ERROR and d.reason == "TEMPLATE_UNUSABLE"


# ── enrolment ────────────────────────────────────────────────────────────

def test_enrolment_collects_required_samples():
    eng = Engine(policy(enrol_samples=4), "ENROL")
    st, _ = eng.start(T0)
    ops = Ops(1)
    d = run(eng, st, ops)
    assert d.outcome == OutcomeEnum.ENROLMENT_CAPTURED
    assert len(st.samples) == 4
    assert all(np.allclose(s, fake_identity_vector(1)) for s in st.samples)


def test_enrolment_samples_reset_on_failed_round():
    eng = Engine(policy(enrol_samples=3, max_attempts=3), "ENROL")
    st, _ = eng.start(T0)
    st.steps = [LOOK_STRAIGHT, TURN_LEFT, LOOK_STRAIGHT]
    ops = Ops(1)
    now = _advance_opening(eng, st, ops, T0)
    assert len(st.samples) == 1
    eng.process(st, frame(face(yaw=-25)), now + timedelta(milliseconds=400), ops)  # wrong way
    assert st.samples == [] and st.round == 1


def test_state_round_trips_through_json():
    from backend.services.biometric.engine import EngineState
    eng = Engine(policy(), "VERIFY")
    st, _ = eng.start(T0)
    eng.process(st, frame(face()), T0 + timedelta(seconds=1), Ops(1))
    clone = EngineState.from_json(st.to_json())
    assert clone.steps == st.steps and clone.hold == st.hold and clone.samples == []
