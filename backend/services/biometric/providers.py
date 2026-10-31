"""Inference providers.

`DeepFaceProvider` is the only provider used in development and production:
  * MediaPipe FaceLandmarker — face detection/count, 478 landmarks, head pose
    (from the facial transformation matrix) and eye-blink blendshapes.
  * DeepFace "Fasnet" (MiniFASNetV2 + V1SE, ONNX) — passive anti-spoofing.
  * DeepFace ArcFace (ONNX) — 512-d embedding of an eye-aligned face crop.
All arrays handed to DeepFace are BGR, which is what DeepFace expects.

`FakeProvider` exists only for isolated automated tests (config refuses it
unless APP_ENV=test). It decodes a synthetic marker grid drawn into test
frames, so browser and API tests can exercise the full pipeline deterministically.
"""
from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from backend.services.biometric.analysis import FaceObservation, FrameAnalysis, PassiveResult

log = logging.getLogger(__name__)

EYE_A = (33, 133)      # eye on the image-left side
EYE_B = (362, 263)     # eye on the image-right side


class InferenceProvider(Protocol):
    name: str
    recognition_model: str
    embedding_dim: int

    def analyze(self, bgr: np.ndarray) -> FrameAnalysis: ...
    def passive_liveness(self, bgr: np.ndarray, face: FaceObservation) -> PassiveResult: ...
    def embed(self, bgr: np.ndarray, face: FaceObservation) -> np.ndarray: ...
    def warmup(self) -> list[dict[str, Any]]: ...


def _crop_stats(bgr: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[float, float]:
    H, W = bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0, 0.0
    gray = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    scale = 160.0 / gray.shape[1]
    gray = cv2.resize(gray, (160, max(8, int(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return min(1.0, lap / 120.0), float(gray.mean()) / 255.0


def aligned_face_crop(bgr: np.ndarray, face: FaceObservation, margin: float = 0.06) -> np.ndarray:
    """Rotate so the eyes are level, then crop the face box (BGR)."""
    (lx, ly), (rx, ry) = face.eye_left_px, face.eye_right_px
    angle = math.degrees(math.atan2(ry - ly, rx - lx))
    H, W = bgr.shape[:2]
    cx_e, cy_e = (lx + rx) / 2.0, (ly + ry) / 2.0
    M = cv2.getRotationMatrix2D((cx_e, cy_e), angle, 1.0)
    rotated = cv2.warpAffine(bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    x, y, w, h = face.bbox_px
    cx, cy = x + w / 2.0, y + h / 2.0
    ncx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    ncy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
    hw, hh = w * (0.5 + margin), h * (0.5 + margin)
    x0, y0 = int(max(0, ncx - hw)), int(max(0, ncy - hh))
    x1, y1 = int(min(W, ncx + hw)), int(min(H, ncy + hh))
    crop = rotated[y0:y1, x0:x1]
    if crop.size == 0 or min(crop.shape[:2]) < 24:
        raise ValueError("Face crop too small")
    return np.ascontiguousarray(crop)


class DeepFaceProvider:
    name = "deepface"

    def __init__(self, landmarker_path: Path, recognition_model: str = "ArcFace") -> None:
        self.recognition_model = recognition_model
        self.embedding_dim = 512
        self._landmarker_path = Path(landmarker_path)
        self._local = threading.local()
        self._load_lock = threading.Lock()
        self._fasnet = None
        self._recognizer = None

    # ── model loading ──────────────────────────────────────────────────────
    def _landmarker(self):
        lm = getattr(self._local, "landmarker", None)
        if lm is None:
            if not self._landmarker_path.exists():
                raise RuntimeError(
                    f"Face landmarker model not found at {self._landmarker_path}. Run: python scripts/fetch_models.py")
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options as bo
            opts = vision.FaceLandmarkerOptions(
                base_options=bo.BaseOptions(model_asset_path=str(self._landmarker_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=4,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            lm = vision.FaceLandmarker.create_from_options(opts)
            self._local.landmarker = lm
        return lm

    def _get_fasnet(self):
        if self._fasnet is None:
            with self._load_lock:
                if self._fasnet is None:
                    from deepface import DeepFace
                    self._fasnet = DeepFace.build_model(model_name="Fasnet", task="spoofing")
        return self._fasnet

    def _get_recognizer(self):
        if self._recognizer is None:
            with self._load_lock:
                if self._recognizer is None:
                    from deepface import DeepFace
                    self._recognizer = DeepFace.build_model(model_name=self.recognition_model)
                    self.embedding_dim = int(getattr(self._recognizer, "output_shape", 512))
        return self._recognizer

    def warmup(self) -> list[dict[str, Any]]:
        results = []
        for name, task, fn in (
            ("MediaPipe FaceLandmarker", "landmarks", self._warm_landmarker),
            ("MiniFASNet (Fasnet)", "antispoof", self._warm_fasnet),
            (self.recognition_model, "recognition", self._warm_recognizer),
        ):
            t0 = time.perf_counter()
            try:
                info = fn() or {}
                results.append({"name": name, "task": task, "status": "READY",
                                "load_ms": (time.perf_counter() - t0) * 1000, **info})
            except Exception as exc:  # noqa: BLE001
                log.exception("Model warm-up failed: %s", name)
                results.append({"name": name, "task": task, "status": "FAILED",
                                "load_ms": (time.perf_counter() - t0) * 1000, "error": f"{type(exc).__name__}: {exc}"[:300]})
        return results

    def _warm_landmarker(self) -> dict[str, Any]:
        lm = self._landmarker()
        import mediapipe as mp
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=blank))
        digest = hashlib.sha256(self._landmarker_path.read_bytes()).hexdigest()
        return {"file_name": self._landmarker_path.name, "sha256": digest}

    def _warm_fasnet(self) -> dict[str, Any]:
        model = self._get_fasnet()
        dummy = np.full((240, 240, 3), 128, dtype=np.uint8)
        model.analyze(img=dummy, facial_area=(60, 60, 120, 120))
        return {"file_name": "2.7_80x80_MiniFASNetV2 + 4_0_0_80x80_MiniFASNetV1SE"}

    def _warm_recognizer(self) -> dict[str, Any]:
        self._get_recognizer()
        from deepface import DeepFace
        dummy = np.full((112, 112, 3), 128, dtype=np.uint8)
        out = DeepFace.represent(img_path=dummy, model_name=self.recognition_model,
                                 detector_backend="skip", enforce_detection=False, align=False)
        self.embedding_dim = len(out[0]["embedding"])
        return {"embedding_dim": self.embedding_dim}

    # ── inference ──────────────────────────────────────────────────────────
    def analyze(self, bgr: np.ndarray) -> FrameAnalysis:
        import mediapipe as mp
        t0 = time.perf_counter()
        H, W = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self._landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        faces: list[FaceObservation] = []
        for i, pts in enumerate(res.face_landmarks):
            xs = np.array([p.x for p in pts])
            ys = np.array([p.y for p in pts])
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            m = np.array(res.facial_transformation_matrixes[i]) if res.facial_transformation_matrixes else np.eye(4)
            yaw = math.degrees(math.atan2(-m[2, 0], math.hypot(m[2, 1], m[2, 2])))
            pitch = math.degrees(math.atan2(m[2, 1], m[2, 2]))
            ea = (float(np.mean([pts[j].x for j in EYE_A])) * W, float(np.mean([pts[j].y for j in EYE_A])) * H)
            eb = (float(np.mean([pts[j].x for j in EYE_B])) * W, float(np.mean([pts[j].y for j in EYE_B])) * H)
            roll = math.degrees(math.atan2(eb[1] - ea[1], eb[0] - ea[0]))
            bs = {c.category_name: float(c.score) for c in res.face_blendshapes[i]} if res.face_blendshapes else {}
            px = (int(x0 * W), int(y0 * H), int((x1 - x0) * W), int((y1 - y0) * H))
            sharp, bright = _crop_stats(bgr, *px)
            faces.append(FaceObservation(
                bbox=(x0, y0, x1 - x0, y1 - y0), yaw=yaw, pitch=pitch, roll=roll,
                blink_left=bs.get("eyeBlinkLeft", 0.0), blink_right=bs.get("eyeBlinkRight", 0.0),
                sharpness=sharp, brightness=bright, bbox_px=px, eye_left_px=ea, eye_right_px=eb,
            ))
        return FrameAnalysis(width=W, height=H, faces=faces, analyze_ms=(time.perf_counter() - t0) * 1000)

    def passive_liveness(self, bgr: np.ndarray, face: FaceObservation) -> PassiveResult:
        t0 = time.perf_counter()
        is_real, score = self._get_fasnet().analyze(img=bgr, facial_area=face.bbox_px)
        return PassiveResult(label="REAL" if is_real else "SPOOF", confidence=float(score),
                             ms=(time.perf_counter() - t0) * 1000)

    def embed(self, bgr: np.ndarray, face: FaceObservation) -> np.ndarray:
        from deepface import DeepFace
        self._get_recognizer()
        crop = aligned_face_crop(bgr, face)
        out = DeepFace.represent(img_path=crop, model_name=self.recognition_model,
                                 detector_backend="skip", enforce_detection=False, align=False)
        vec = np.asarray(out[0]["embedding"], dtype=np.float64)
        if vec.shape[0] != self.embedding_dim:
            raise ValueError(f"Unexpected embedding size {vec.shape[0]}")
        return vec


# ── Test-only provider ─────────────────────────────────────────────────────

CELL = 40           # px; marker cells are drawn in the top-left row of the frame
MAGIC = 10
N_CELLS = 8


def _cell_value(bgr: np.ndarray, idx: int) -> int:
    pad = 8
    patch = bgr[pad:CELL - pad, idx * CELL + pad:(idx + 1) * CELL - pad]
    if patch.size == 0:
        return -1
    return int(round(float(patch.mean()) / 17.0))


def _base_vector(identity: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + identity)
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


# Codes 13–15 are "near" variants of identity 1 at known cosine distances, so
# threshold behaviour can be tested deterministically.
NEAR_IDENTITY_DISTANCE = {13: 0.20, 14: 0.36, 15: 0.55}


def fake_identity_vector(identity: int, dim: int = 512) -> np.ndarray:
    if identity in NEAR_IDENTITY_DISTANCE:
        base = _base_vector(1, dim)
        other = _base_vector(100 + identity, dim)
        other = other - np.dot(other, base) * base       # orthogonal to base
        other /= np.linalg.norm(other)
        cos = 1.0 - NEAR_IDENTITY_DISTANCE[identity]
        v = cos * base + np.sqrt(1 - cos * cos) * other
        return v / np.linalg.norm(v)
    return _base_vector(identity, dim)


class FakeProvider:
    """Deterministic provider for isolated tests (APP_ENV=test only).

    Marker cells (value v in 0..15, gray = 17·v):
      0 magic(=10)  1 face count  2 yaw=(v-8)·5°  3 blink=v/15
      4 identity    5 real(v≥8)   6 quality=v/15 7 face width=0.08+0.04·v
    """

    name = "fake"
    recognition_model = "ArcFace"
    embedding_dim = 512

    def warmup(self) -> list[dict[str, Any]]:
        return [{"name": f"FakeProvider {task} (test only)", "task": task, "status": "READY", "load_ms": 0.0}
                for task in ("landmarks", "antispoof", "recognition")]

    def _codes(self, bgr: np.ndarray) -> list[int]:
        if bgr.shape[0] < CELL or bgr.shape[1] < CELL * N_CELLS:
            return []
        return [_cell_value(bgr, i) for i in range(N_CELLS)]

    def analyze(self, bgr: np.ndarray) -> FrameAnalysis:
        H, W = bgr.shape[:2]
        c = self._codes(bgr)
        if not c or c[0] != MAGIC:
            return FrameAnalysis(width=W, height=H, faces=[])
        count = max(0, min(3, c[1]))
        width = min(0.9, 0.08 + 0.04 * c[7])
        q = c[6] / 15.0
        faces = []
        for k in range(count):
            cx = 0.5 + (k * 0.3 if k else 0.0) - (0.15 if count > 1 else 0.0)
            x, y = cx - width / 2, 0.52 - width * 0.65
            h = width * 1.3
            faces.append(FaceObservation(
                bbox=(x, y, width, h), yaw=(c[2] - 8) * 5.0, pitch=0.0, roll=0.0,
                blink_left=c[3] / 15.0, blink_right=c[3] / 15.0,
                sharpness=q, brightness=0.25 + 0.5 * q,
                bbox_px=(int(x * W), int(y * H), int(width * W), int(h * H)),
                extra={"identity": c[4], "real": c[5] >= 8},
            ))
        return FrameAnalysis(width=W, height=H, faces=faces, analyze_ms=1.0)

    def passive_liveness(self, bgr: np.ndarray, face: FaceObservation) -> PassiveResult:
        real = bool(face.extra.get("real", False))
        return PassiveResult(label="REAL" if real else "SPOOF", confidence=0.9, ms=0.5)

    def embed(self, bgr: np.ndarray, face: FaceObservation) -> np.ndarray:
        return fake_identity_vector(int(face.extra.get("identity", 0)), self.embedding_dim)


def make_fake_frame(face_count: int = 1, yaw_deg: float = 0.0, blink: float = 0.0, identity: int = 1,
                    real: bool = True, quality: float = 0.9, width_code: int = 7,
                    nonce: int = 0, size: tuple[int, int] = (640, 480)) -> bytes:
    """Encode a synthetic test frame understood by FakeProvider (JPEG bytes)."""
    w, h = size
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    # vary the body so consecutive frames are never byte-identical
    rng = np.random.default_rng(nonce)
    img[CELL + 10:, :] = rng.integers(60, 140, size=(h - CELL - 10, w, 3), dtype=np.uint8)
    vals = [MAGIC, face_count, int(round(yaw_deg / 5.0)) + 8, int(round(blink * 15)), identity,
            12 if real else 2, int(round(quality * 15)), width_code]
    for i, v in enumerate(vals):
        img[0:CELL, i * CELL:(i + 1) * CELL] = max(0, min(15, v)) * 17
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buf.tobytes()
