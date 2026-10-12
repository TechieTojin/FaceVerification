"""
FaceVerification v1.3
Real-time face verification with async inference, anti-spoofing, and state machine.

Usage:
    python face_verification.py [--reference PATH] [--camera INDEX]
    python face_verification.py --help

Configuration via environment variables or a .env file:
    REFERENCE_IMAGE_PATH        Path to the reference face image
    CAMERA_INDEX                Camera device index (default: 0)
    CAMERA_WIDTH / CAMERA_HEIGHT  Resolution (default: 640x480)
    RECOGNITION_MODEL           DeepFace model (default: ArcFace)
    DETECTOR_BACKEND            Face detector (default: opencv)
    ENABLE_LIVENESS             Anti-spoof check (default: true)
    ENABLE_PERFORMANCE_OVERLAY  Show FPS/inference HUD (default: true)
    INFERENCE_INTERVAL_MS       Time between inferences (default: 500)
    RESULT_EXPIRY_MS            VERIFIED state lifetime (default: 1500)
    LOG_LEVEL                   Logging verbosity (default: INFO)
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from app.config import AppConfig
from app.logging_config import setup_logging
from app.camera import CameraManager
from app.state_manager import StateManager, VerificationResult, VerificationState
from app.inference_worker import InferenceWorker
from app.overlay import draw_overlay

WINDOW_NAME = "FaceVerification"

# ── v1.1 compatibility shims (kept for existing tests) ─────────────────────────

def load_config(args: argparse.Namespace) -> dict:
    """v1.1 compat: merge CLI args with environment/dotenv values."""
    ref_path = args.reference or os.getenv("REFERENCE_IMAGE_PATH", "")
    camera_index = args.camera if args.camera is not None else int(os.getenv("CAMERA_INDEX", "0"))
    model = os.getenv("FACE_MODEL", "ArcFace")
    detector = os.getenv("DETECTOR_BACKEND", "opencv")
    return {
        "reference_path": ref_path,
        "camera_index": camera_index,
        "model": model,
        "detector": detector,
    }


def open_camera(index: int) -> cv2.VideoCapture:
    """v1.1 compat: open the webcam and configure resolution."""
    import sys as _sys
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logging.error(
            "Cannot open camera (index %d).\n"
            "  Check that the camera is connected and not used by another application.\n"
            "  Try a different index with --camera 1",
            index,
        )
        _sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info("Camera %d opened at %dx%d", index, actual_w, actual_h)
    return cap


def verify_face(frame, reference, model: str, detector: str):
    """v1.1 compat: run DeepFace verification."""
    from deepface import DeepFace  # noqa: PLC0415
    try:
        result = DeepFace.verify(
            frame,
            reference.copy(),
            model_name=model,
            detector_backend=detector,
            enforce_detection=True,
        )
        return bool(result["verified"])
    except ValueError:
        return False
    except Exception as exc:  # noqa: BLE001
        logging.warning("Verification error: %s", exc)
        return None


def draw_overlay(frame, match, fps: float, last_inference_ms: float) -> None:
    """v1.1 compat: draw MATCH / NOT A MATCH overlays."""
    FONT = cv2.FONT_HERSHEY_DUPLEX
    COLOR_MATCH = (0, 220, 0)
    COLOR_NO_MATCH = (0, 0, 220)
    COLOR_INFO = (220, 220, 220)
    COLOR_WARN = (0, 165, 255)
    h, w = frame.shape[:2]
    if match is True:
        label, color = "MATCH", COLOR_MATCH
    elif match is False:
        label, color = "NOT A MATCH", COLOR_NO_MATCH
    else:
        label, color = "CHECKING...", COLOR_WARN
    cv2.putText(frame, label, (20, h - 30), FONT, 1.2, color, 2)
    hud = f"FPS: {fps:.1f}  |  Inference: {last_inference_ms:.0f}ms  |  Press Q to quit"
    cv2.putText(frame, hud, (10, 22), FONT, 0.5, COLOR_INFO, 1)


# ── end v1.1 shims ─────────────────────────────────────────────────────────────


def load_reference_image(path: str) -> np.ndarray:
    """Load and validate the reference face image; exits on failure."""
    if not path:
        logging.error(
            "No reference image path provided.\n"
            "  Set REFERENCE_IMAGE_PATH in .env or pass --reference <path>"
        )
        sys.exit(1)
    p = Path(path)
    if not p.exists():
        logging.error("Reference image not found: %s", path)
        sys.exit(1)
    img = cv2.imread(str(p))
    if img is None or img.size == 0:
        logging.error("Could not read reference image: %s", path)
        sys.exit(1)
    logging.info("Reference image loaded: %s  (%dx%d)", p.name, img.shape[1], img.shape[0])
    return img


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FaceVerification v1.3 — real-time face verification with anti-spoofing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--reference", "-r",
        metavar="PATH",
        help="Path to the reference face image (overrides REFERENCE_IMAGE_PATH env var)",
    )
    p.add_argument(
        "--camera", "-c",
        type=int,
        metavar="INDEX",
        help="Camera device index (default: 0)",
    )
    return p.parse_args()


def run(config: AppConfig) -> None:
    """Main application loop."""
    log = logging.getLogger(__name__)

    reference = load_reference_image(config.reference_image_path)

    camera_mgr = CameraManager(
        index=config.camera_index,
        width=config.camera_width,
        height=config.camera_height,
    )
    cap = camera_mgr.open()

    worker = InferenceWorker(
        reference=reference,
        model_name=config.recognition_model,
        detector_backend=config.detector_backend,
        enable_liveness=config.enable_liveness,
        inference_interval_ms=config.inference_interval_ms,
    )
    state_mgr = StateManager(result_expiry_ms=config.result_expiry_ms)

    log.info(
        "Starting — model=%s detector=%s liveness=%s  |  Press Q/ESC to quit",
        config.recognition_model,
        config.detector_backend,
        config.enable_liveness,
    )

    # Transition to SCANNING before starting worker
    state_mgr.update(VerificationResult(
        state=VerificationState.SCANNING,
        timestamp=time.monotonic(),
    ))

    worker.start()

    frame_count = 0
    fps_start = time.monotonic()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to read frame — skipping")
                time.sleep(0.02)
                continue

            # Push frame to worker (non-blocking)
            worker.push_frame(frame)
            frame_count += 1

            # Drain latest result from worker queue
            try:
                result = worker.result_queue.get_nowait()
                state_mgr.update(result)
            except queue.Empty:
                pass

            # Expiry tick
            state_mgr.tick()

            # FPS
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.monotonic()

            # Draw and display
            draw_overlay(
                frame,
                state_mgr.current_result,
                fps,
                enable_performance_overlay=config.enable_performance_overlay,
                enable_liveness=config.enable_liveness,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                log.info("Exit key pressed — shutting down")
                break

    finally:
        worker.stop()
        camera_mgr.release()
        cv2.destroyAllWindows()
        log.info("Goodbye.")


if __name__ == "__main__":
    args = parse_args()
    config = AppConfig.from_env()
    config.apply_cli_overrides(reference=args.reference, camera=args.camera)
    setup_logging(config.log_level)
    run(config)
