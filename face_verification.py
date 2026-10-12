"""
FaceVerification v1.1
Real-time face verification using a webcam and a reference image.

Usage:
    python face_verification.py [--reference PATH] [--camera INDEX]
    python face_verification.py --help

Configuration can also be set via environment variables or a .env file:
    REFERENCE_IMAGE_PATH   Path to the reference face image
    CAMERA_INDEX           Camera device index (default: 0)
    FACE_MODEL             DeepFace model (default: ArcFace)
    DETECTOR_BACKEND       Face detector (default: opencv)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
WINDOW_NAME = "FaceVerification"
FRAME_W = 640
FRAME_H = 480
FONT = cv2.FONT_HERSHEY_DUPLEX
COLOR_MATCH = (0, 220, 0)
COLOR_NO_MATCH = (0, 0, 220)
COLOR_INFO = (220, 220, 220)
COLOR_WARN = (0, 165, 255)


# ── configuration ──────────────────────────────────────────────────────────────

def load_config(args: argparse.Namespace) -> dict:
    """Merge CLI args with environment/dotenv values."""
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


# ── reference image ────────────────────────────────────────────────────────────

def load_reference_image(path: str) -> np.ndarray:
    """Load and validate the reference face image.

    Raises SystemExit with a clear message on any failure.
    """
    if not path:
        log.error(
            "No reference image path provided.\n"
            "  Set REFERENCE_IMAGE_PATH in .env or pass --reference <path>"
        )
        sys.exit(1)

    p = Path(path)
    if not p.exists():
        log.error("Reference image not found: %s", path)
        sys.exit(1)

    img = cv2.imread(str(p))
    if img is None:
        log.error(
            "Could not read reference image (unsupported format or corrupt file): %s", path
        )
        sys.exit(1)

    if img.size == 0:
        log.error("Reference image is empty: %s", path)
        sys.exit(1)

    log.info("Reference image loaded: %s  (%dx%d)", p.name, img.shape[1], img.shape[0])
    return img


# ── camera ─────────────────────────────────────────────────────────────────────

def open_camera(index: int) -> cv2.VideoCapture:
    """Open the webcam and configure resolution.

    Raises SystemExit with a clear message if the camera cannot be opened.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        log.error(
            "Cannot open camera (index %d).\n"
            "  Check that the camera is connected and not used by another application.\n"
            "  Try a different index with --camera 1",
            index,
        )
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Camera %d opened at %dx%d", index, actual_w, actual_h)
    return cap


# ── face verification ──────────────────────────────────────────────────────────

def verify_face(
    frame: np.ndarray,
    reference: np.ndarray,
    model: str,
    detector: str,
) -> Optional[bool]:
    """Run DeepFace verification between frame and reference image.

    Returns:
        True   — faces match
        False  — faces do not match or no face detected
        None   — unexpected error (logged; does not crash app)
    """
    # Import here so DeepFace's TF initialisation only happens when needed
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
        # No face detected in one or both images — expected, not an error
        return False
    except Exception as exc:  # noqa: BLE001
        log.warning("Verification error: %s", exc)
        return None


# ── overlay drawing ────────────────────────────────────────────────────────────

def draw_overlay(
    frame: np.ndarray,
    match: Optional[bool],
    fps: float,
    last_inference_ms: float,
) -> None:
    """Draw MATCH / NOT A MATCH / CHECKING overlays and HUD info."""
    h, w = frame.shape[:2]

    # Result banner
    if match is True:
        label = "MATCH"
        color = COLOR_MATCH
    elif match is False:
        label = "NOT A MATCH"
        color = COLOR_NO_MATCH
    else:
        label = "CHECKING..."
        color = COLOR_WARN

    cv2.putText(frame, label, (20, h - 30), FONT, 1.2, color, 2)

    # HUD: FPS and inference time
    hud = f"FPS: {fps:.1f}  |  Inference: {last_inference_ms:.0f}ms  |  Press Q to quit"
    cv2.putText(frame, hud, (10, 22), FONT, 0.5, COLOR_INFO, 1)


# ── main loop ──────────────────────────────────────────────────────────────────

def run(config: dict) -> None:
    """Main application entry point."""
    reference = load_reference_image(config["reference_path"])
    cap = open_camera(config["camera_index"])

    model = config["model"]
    detector = config["detector"]

    log.info("Model: %s  |  Detector: %s", model, detector)
    log.info("Starting — press Q in the video window to quit")

    face_match: Optional[bool] = None
    last_inference_ms: float = 0.0
    frame_count = 0
    fps_start = time.monotonic()
    fps = 0.0

    # Throttle: run inference at most once every N frames to keep UI responsive
    INFERENCE_EVERY_N_FRAMES = 5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to read frame from camera — skipping")
                time.sleep(0.05)
                continue

            frame_count += 1

            # Run inference periodically (not every frame) to keep UI fluid
            if frame_count % INFERENCE_EVERY_N_FRAMES == 0:
                t0 = time.monotonic()
                face_match = verify_face(frame, reference, model, detector)
                last_inference_ms = (time.monotonic() - t0) * 1000

            # FPS counter
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.monotonic()

            draw_overlay(frame, face_match, fps, last_inference_ms)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                log.info("Exit key pressed — shutting down")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        log.info("Camera released. Goodbye.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FaceVerification v1.1 — real-time face verification",
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


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    run(config)
