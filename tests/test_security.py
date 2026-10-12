"""Security state machine tests — fail-closed guarantees."""
from __future__ import annotations

import time

import pytest

from app.state_manager import StateManager, VerificationResult, VerificationState


def _result(state: VerificationState, age_s: float = 0.0) -> VerificationResult:
    return VerificationResult(
        state=state,
        timestamp=time.monotonic() - age_s,
    )


NON_AUTH_STATES = [
    VerificationState.INITIALISING,
    VerificationState.READY,
    VerificationState.SCANNING,
    VerificationState.VERIFYING,
    VerificationState.REJECTED,
    VerificationState.NO_FACE,
    VerificationState.MULTIPLE_FACES,
    VerificationState.SPOOF_DETECTED,
    VerificationState.ERROR,
]


class TestNeverAuthenticated:
    @pytest.mark.parametrize("state", NON_AUTH_STATES)
    def test_non_verified_state_never_authenticated(self, state: VerificationState):
        mgr = StateManager(result_expiry_ms=5000)
        mgr.update(_result(state))
        assert mgr.is_authenticated() is False, f"State {state} must never authenticate"

    def test_spoof_detected_never_authenticated(self):
        mgr = StateManager(result_expiry_ms=5000)
        mgr.update(_result(VerificationState.SPOOF_DETECTED))
        assert mgr.is_authenticated() is False

    def test_error_never_authenticated(self):
        mgr = StateManager(result_expiry_ms=5000)
        mgr.update(_result(VerificationState.ERROR))
        assert mgr.is_authenticated() is False

    def test_multiple_faces_never_authenticated(self):
        mgr = StateManager(result_expiry_ms=5000)
        mgr.update(_result(VerificationState.MULTIPLE_FACES))
        assert mgr.is_authenticated() is False

    def test_no_face_never_authenticated(self):
        mgr = StateManager(result_expiry_ms=5000)
        mgr.update(_result(VerificationState.NO_FACE))
        assert mgr.is_authenticated() is False

    def test_expired_verified_never_authenticated(self):
        mgr = StateManager(result_expiry_ms=50)
        # Construct a result that is already past expiry
        mgr.update(_result(VerificationState.VERIFIED, age_s=0.2))
        assert mgr.is_authenticated() is False

    def test_only_fresh_verified_authenticated(self):
        mgr = StateManager(result_expiry_ms=1000)
        mgr.update(_result(VerificationState.VERIFIED))
        assert mgr.is_authenticated() is True


class TestStateTransitions:
    def test_verified_after_expiry_becomes_not_authenticated(self):
        mgr = StateManager(result_expiry_ms=80)
        mgr.update(_result(VerificationState.VERIFIED))
        assert mgr.is_authenticated() is True
        time.sleep(0.12)
        assert mgr.is_authenticated() is False

    def test_tick_clears_expired_verified(self):
        mgr = StateManager(result_expiry_ms=50)
        mgr.update(_result(VerificationState.VERIFIED))
        time.sleep(0.1)
        mgr.tick()
        assert mgr.state == VerificationState.SCANNING
        assert mgr.is_authenticated() is False

    def test_repeated_rejected_never_flips_to_verified(self):
        mgr = StateManager(result_expiry_ms=1000)
        for _ in range(20):
            mgr.update(_result(VerificationState.REJECTED))
            mgr.tick()
            assert mgr.is_authenticated() is False
