"""Tests for StateManager and VerificationState."""
from __future__ import annotations

import time

import pytest

from app.state_manager import StateManager, VerificationResult, VerificationState


def _result(state: VerificationState, age_s: float = 0.0) -> VerificationResult:
    return VerificationResult(
        state=state,
        timestamp=time.monotonic() - age_s,
    )


class TestStateManagerExpiry:
    def test_verified_expires_after_expiry_time(self):
        mgr = StateManager(result_expiry_ms=100)
        mgr.update(_result(VerificationState.VERIFIED))
        assert mgr.state == VerificationState.VERIFIED
        time.sleep(0.15)
        mgr.tick()
        assert mgr.state == VerificationState.SCANNING

    def test_verified_does_not_expire_before_timeout(self):
        mgr = StateManager(result_expiry_ms=500)
        mgr.update(_result(VerificationState.VERIFIED))
        time.sleep(0.05)
        mgr.tick()
        assert mgr.state == VerificationState.VERIFIED

    def test_non_verified_states_not_changed_by_tick(self):
        for state in [
            VerificationState.REJECTED,
            VerificationState.NO_FACE,
            VerificationState.SCANNING,
            VerificationState.ERROR,
            VerificationState.SPOOF_DETECTED,
            VerificationState.MULTIPLE_FACES,
        ]:
            mgr = StateManager(result_expiry_ms=100)
            mgr.update(_result(state))
            time.sleep(0.15)
            mgr.tick()
            assert mgr.state == state, f"Expected {state} to be unchanged, got {mgr.state}"

    def test_expired_verified_transitions_to_scanning(self):
        mgr = StateManager(result_expiry_ms=50)
        mgr.update(_result(VerificationState.VERIFIED))
        time.sleep(0.1)
        mgr.tick()
        assert mgr.state == VerificationState.SCANNING


class TestIsAuthenticated:
    def test_false_for_rejected(self):
        mgr = StateManager()
        mgr.update(_result(VerificationState.REJECTED))
        assert mgr.is_authenticated() is False

    def test_false_for_no_face(self):
        mgr = StateManager()
        mgr.update(_result(VerificationState.NO_FACE))
        assert mgr.is_authenticated() is False

    def test_false_for_error(self):
        mgr = StateManager()
        mgr.update(_result(VerificationState.ERROR))
        assert mgr.is_authenticated() is False

    def test_false_for_spoof_detected(self):
        mgr = StateManager()
        mgr.update(_result(VerificationState.SPOOF_DETECTED))
        assert mgr.is_authenticated() is False

    def test_false_for_scanning(self):
        mgr = StateManager()
        mgr.update(_result(VerificationState.SCANNING))
        assert mgr.is_authenticated() is False

    def test_false_for_multiple_faces(self):
        mgr = StateManager()
        mgr.update(_result(VerificationState.MULTIPLE_FACES))
        assert mgr.is_authenticated() is False

    def test_false_for_initialising(self):
        mgr = StateManager()
        assert mgr.is_authenticated() is False

    def test_true_for_fresh_verified(self):
        mgr = StateManager(result_expiry_ms=1000)
        mgr.update(_result(VerificationState.VERIFIED))
        assert mgr.is_authenticated() is True

    def test_false_for_expired_verified(self):
        mgr = StateManager(result_expiry_ms=50)
        mgr.update(_result(VerificationState.VERIFIED, age_s=0.1))
        assert mgr.is_authenticated() is False
