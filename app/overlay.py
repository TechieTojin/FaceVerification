"""Draw state overlays on camera frames."""
from __future__ import annotations

import cv2
import numpy as np

from app.state_manager import VerificationResult, VerificationState

FONT = cv2.FONT_HERSHEY_DUPLEX

# Color map per state (BGR)
_STATE_COLORS: dict[VerificationState, tuple[int, int, int]] = {
    VerificationState.VERIFIED: (0, 220, 0),          # green
    VerificationState.REJECTED: (0, 0, 220),           # red
    VerificationState.SPOOF_DETECTED: (0, 0, 220),     # red
    VerificationState.NO_FACE: (0, 200, 220),          # yellow
    VerificationState.MULTIPLE_FACES: (0, 128, 255),   # orange
    VerificationState.ERROR: (220, 0, 220),            # magenta
}
_DEFAULT_COLOR = (220, 220, 220)  # white-ish


def _color(state: VerificationState) -> tuple[int, int, int]:
    return _STATE_COLORS.get(state, _DEFAULT_COLOR)


def draw_overlay(
    frame: np.ndarray,
    result: VerificationResult,
    fps: float,
    enable_performance_overlay: bool = True,
    enable_liveness: bool = True,
) -> None:
    """Mutate *frame* in place with state info and optional performance HUD."""
    h, w = frame.shape[:2]
    state = result.state
    color = _color(state)

    # State label
    cv2.putText(frame, state.value, (20, h - 50), FONT, 1.1, color, 2)

    # AUTHENTICATED banner
    if state == VerificationState.VERIFIED:
        cv2.putText(
            frame, "AUTHENTICATED", (20, h - 85), FONT, 0.9, color, 2
        )

    # Liveness indicator
    if enable_liveness:
        liveness_text = f"Liveness: {result.liveness_score:.2f}"
        cv2.putText(frame, liveness_text, (20, 50), FONT, 0.5, _DEFAULT_COLOR, 1)

    # Performance HUD
    if enable_performance_overlay:
        hud = (
            f"FPS: {fps:.1f}  |  "
            f"Inference: {result.inference_ms:.0f}ms  |  "
            f"Faces: {result.face_count}  |  "
            "Press Q to quit"
        )
        cv2.putText(frame, hud, (10, 22), FONT, 0.45, _DEFAULT_COLOR, 1)
