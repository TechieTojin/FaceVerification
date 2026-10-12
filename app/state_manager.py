"""VerificationState enum, VerificationResult dataclass, and StateManager."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VerificationState(Enum):
    INITIALISING = "INITIALISING"
    READY = "READY"
    SCANNING = "SCANNING"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    SPOOF_DETECTED = "SPOOF_DETECTED"
    ERROR = "ERROR"


@dataclass
class VerificationResult:
    state: VerificationState
    timestamp: float  # time.monotonic()
    distance: float = 0.0
    threshold: float = 0.0
    liveness_score: float = 0.0
    inference_ms: float = 0.0
    face_count: int = 0


class StateManager:
    """Holds the current VerificationResult and handles VERIFIED expiry.

    Fail-closed: is_authenticated() returns True ONLY when state is VERIFIED
    AND the result has not expired.
    """

    def __init__(self, result_expiry_ms: float = 1500.0) -> None:
        self._expiry_s: float = result_expiry_ms / 1000.0
        self._result: VerificationResult = VerificationResult(
            state=VerificationState.INITIALISING,
            timestamp=time.monotonic(),
        )

    @property
    def current_result(self) -> VerificationResult:
        return self._result

    @property
    def state(self) -> VerificationState:
        return self._result.state

    def update(self, result: VerificationResult) -> None:
        """Replace the current result."""
        self._result = result

    def tick(self) -> None:
        """Check expiry — if VERIFIED and expired, transition to SCANNING."""
        if self._result.state == VerificationState.VERIFIED:
            age = time.monotonic() - self._result.timestamp
            if age >= self._expiry_s:
                self._result = VerificationResult(
                    state=VerificationState.SCANNING,
                    timestamp=time.monotonic(),
                )

    def is_authenticated(self) -> bool:
        """Return True ONLY for a fresh VERIFIED result (fail-closed)."""
        if self._result.state != VerificationState.VERIFIED:
            return False
        age = time.monotonic() - self._result.timestamp
        return age < self._expiry_s
