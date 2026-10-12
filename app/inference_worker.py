"""Background inference thread: reads latest frame, runs DeepFace pipeline."""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

from app.state_manager import VerificationResult, VerificationState

log = logging.getLogger(__name__)


class InferenceWorker:
    """Single background thread that processes the latest available frame.

    Design:
    - Latest frame is stored under a lock (only the most recent frame matters).
    - Results are placed in a maxsize=1 queue (only latest result matters).
    - The display loop is never blocked by DeepFace inference.
    - All security decisions are fail-closed (exception → ERROR, never VERIFIED).
    """

    def __init__(
        self,
        reference: np.ndarray,
        model_name: str = "ArcFace",
        detector_backend: str = "opencv",
        enable_liveness: bool = True,
        inference_interval_ms: float = 500.0,
    ) -> None:
        self._reference = reference
        self._model_name = model_name
        self._detector_backend = detector_backend
        self._enable_liveness = enable_liveness
        self._interval_s = inference_interval_ms / 1000.0

        # Latest frame storage
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()

        # Output: maxsize=1 — only keep latest result
        self.result_queue: queue.Queue[VerificationResult] = queue.Queue(maxsize=1)

        # Control
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="InferenceWorker")

    # ── public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread.start()
        log.info("InferenceWorker started (liveness=%s)", self._enable_liveness)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        self._frame_event.set()  # wake up the thread if it is waiting
        self._thread.join(timeout=timeout)
        log.info("InferenceWorker stopped")

    def push_frame(self, frame: np.ndarray) -> None:
        """Store the latest frame (thread-safe, non-blocking)."""
        with self._frame_lock:
            self._frame = frame.copy()
        self._frame_event.set()

    # ── internal ───────────────────────────────────────────────────────────────

    def _get_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._frame

    def _put_result(self, result: VerificationResult) -> None:
        """Put result in queue; drop previous if full (keep latest)."""
        try:
            self.result_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.result_queue.put_nowait(result)
        except queue.Full:
            pass

    def _run(self) -> None:
        """Main worker loop."""
        while not self._stop_event.is_set():
            # Wait for a frame or stop signal
            self._frame_event.wait(timeout=self._interval_s)
            self._frame_event.clear()

            if self._stop_event.is_set():
                break

            frame = self._get_frame()
            if frame is None:
                continue

            result = self._process(frame)
            self._put_result(result)

            # Respect inference interval (avoid hammering DeepFace)
            if not self._stop_event.is_set():
                time.sleep(self._interval_s)

    def _process(self, frame: np.ndarray) -> VerificationResult:
        """Run the full detection + liveness + verification pipeline."""
        t0 = time.monotonic()

        # Step 1: validate frame
        if frame is None or frame.size == 0:
            return VerificationResult(
                state=VerificationState.ERROR,
                timestamp=time.monotonic(),
                inference_ms=(time.monotonic() - t0) * 1000,
            )

        try:
            # Lazy import to avoid slow TF init at module load
            from deepface import DeepFace  # noqa: PLC0415

            # Step 2: detect faces (with optional anti-spoofing)
            try:
                if self._enable_liveness:
                    faces = DeepFace.extract_faces(
                        img_path=frame,
                        anti_spoofing=True,
                        detector_backend=self._detector_backend,
                        enforce_detection=False,
                    )
                else:
                    faces = DeepFace.extract_faces(
                        img_path=frame,
                        detector_backend=self._detector_backend,
                        enforce_detection=False,
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("Face extraction failed: %s", exc)
                return VerificationResult(
                    state=VerificationState.ERROR,
                    timestamp=time.monotonic(),
                    inference_ms=(time.monotonic() - t0) * 1000,
                )

            # Step 3: count faces
            face_count = len(faces) if faces else 0
            if face_count == 0:
                return VerificationResult(
                    state=VerificationState.NO_FACE,
                    timestamp=time.monotonic(),
                    face_count=0,
                    inference_ms=(time.monotonic() - t0) * 1000,
                )
            if face_count > 1:
                return VerificationResult(
                    state=VerificationState.MULTIPLE_FACES,
                    timestamp=time.monotonic(),
                    face_count=face_count,
                    inference_ms=(time.monotonic() - t0) * 1000,
                )

            # Step 4: liveness check
            liveness_score = 0.0
            if self._enable_liveness:
                face = faces[0]
                # DeepFace 0.0.102 returns 'is_real' and 'antispoof_score'
                is_real = face.get("is_real", None)
                liveness_score = float(face.get("antispoof_score", 0.0))
                # Fail closed: if we cannot confirm live → SPOOF_DETECTED
                if is_real is None or not is_real:
                    return VerificationResult(
                        state=VerificationState.SPOOF_DETECTED,
                        timestamp=time.monotonic(),
                        liveness_score=liveness_score,
                        face_count=face_count,
                        inference_ms=(time.monotonic() - t0) * 1000,
                    )

            # Step 5: verify identity
            try:
                verify_result = DeepFace.verify(
                    img1_path=frame,
                    img2_path=self._reference.copy(),
                    model_name=self._model_name,
                    detector_backend=self._detector_backend,
                    enforce_detection=True,
                )
            except ValueError:
                # No face detected during verification
                return VerificationResult(
                    state=VerificationState.NO_FACE,
                    timestamp=time.monotonic(),
                    face_count=0,
                    liveness_score=liveness_score,
                    inference_ms=(time.monotonic() - t0) * 1000,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Verification error: %s", exc)
                return VerificationResult(
                    state=VerificationState.ERROR,
                    timestamp=time.monotonic(),
                    liveness_score=liveness_score,
                    face_count=face_count,
                    inference_ms=(time.monotonic() - t0) * 1000,
                )

            # Step 6: decision
            verified = bool(verify_result.get("verified", False))
            distance = float(verify_result.get("distance", 0.0))
            threshold = float(verify_result.get("threshold", 0.0))
            state = VerificationState.VERIFIED if verified else VerificationState.REJECTED

            return VerificationResult(
                state=state,
                timestamp=time.monotonic(),
                distance=distance,
                threshold=threshold,
                liveness_score=liveness_score,
                face_count=face_count,
                inference_ms=(time.monotonic() - t0) * 1000,
            )

        except Exception as exc:  # noqa: BLE001
            log.error("Unexpected inference error: %s", exc)
            return VerificationResult(
                state=VerificationState.ERROR,
                timestamp=time.monotonic(),
                inference_ms=(time.monotonic() - t0) * 1000,
            )
