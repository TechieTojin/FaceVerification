"""Biometric service wrapping DeepFace for face detection, embedding, and comparison."""
from __future__ import annotations
import io
import logging
import time
from typing import Optional

import numpy as np
from PIL import Image
from cryptography.fernet import Fernet

from backend.config import settings
from backend.schemas.enrolment import FaceDetectionResult, QualityResult

log = logging.getLogger(__name__)

# Module-level Fernet key (initialized lazily)
_fernet: Optional[Fernet] = None
_fernet_key: Optional[bytes] = None


def _get_fernet() -> Fernet:
    global _fernet, _fernet_key
    if _fernet is None:
        key = settings.get_fernet_key()
        _fernet_key = key
        _fernet = Fernet(key)
    return _fernet


def image_bytes_to_array(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes (JPEG/PNG) to numpy RGB array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)


def detect_face(image_bytes: bytes) -> FaceDetectionResult:
    """Detect faces in image bytes."""
    try:
        from deepface import DeepFace
        img = image_bytes_to_array(image_bytes)
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend=settings.detector_backend,
            enforce_detection=False,
        )
        face_count = len(faces) if faces else 0
        confidence = float(faces[0].get("confidence", 0.0)) if face_count > 0 else 0.0
        return FaceDetectionResult(
            face_detected=face_count > 0,
            face_count=face_count,
            confidence=confidence,
            message="OK" if face_count == 1 else ("No face" if face_count == 0 else "Multiple faces"),
        )
    except Exception as e:
        log.warning("Face detection error: %s", e)
        return FaceDetectionResult(
            face_detected=False,
            face_count=0,
            confidence=0.0,
            message=f"Error: {str(e)[:100]}",
        )


def assess_quality(image_bytes: bytes) -> QualityResult:
    """Assess image quality for enrolment."""
    try:
        img = image_bytes_to_array(image_bytes)
        # Sharpness: Laplacian variance
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_normalized = min(1.0, sharpness / 500.0)

        # Brightness
        brightness = float(np.mean(gray)) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2.0  # best at 0.5

        # Face size ratio
        from deepface import DeepFace
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend=settings.detector_backend,
            enforce_detection=False,
        )
        face_size_ratio = 0.0
        if faces:
            face = faces[0]
            region = face.get("facial_area", {})
            face_w = region.get("w", 0)
            face_h = region.get("h", 0)
            img_h, img_w = img.shape[:2]
            face_size_ratio = (face_w * face_h) / (img_w * img_h)

        quality_score = (sharpness_normalized * 0.4 + brightness_score * 0.3 + min(1.0, face_size_ratio / 0.1) * 0.3)
        is_acceptable = quality_score >= settings.min_face_quality

        messages = []
        if sharpness_normalized < 0.3:
            messages.append("image is blurry")
        if brightness < 0.2:
            messages.append("too dark")
        elif brightness > 0.9:
            messages.append("too bright")
        if face_size_ratio < 0.05:
            messages.append("face too small, move closer")

        msg = "Quality acceptable" if is_acceptable else f"Quality issues: {', '.join(messages) or 'low score'}"
        return QualityResult(
            quality_score=round(quality_score, 3),
            sharpness=round(sharpness_normalized, 3),
            brightness=round(brightness, 3),
            face_size_ratio=round(face_size_ratio, 3),
            is_acceptable=is_acceptable,
            message=msg,
        )
    except Exception as e:
        log.warning("Quality assessment error: %s", e)
        return QualityResult(
            quality_score=0.0,
            sharpness=0.0,
            brightness=0.0,
            face_size_ratio=0.0,
            is_acceptable=False,
            message=f"Assessment error: {str(e)[:100]}",
        )


def generate_embedding(image_bytes: bytes) -> np.ndarray:
    """Generate face embedding using ArcFace."""
    from deepface import DeepFace
    img = image_bytes_to_array(image_bytes)
    result = DeepFace.represent(
        img_path=img,
        model_name=settings.arcface_model,
        detector_backend=settings.detector_backend,
        enforce_detection=True,
    )
    if not result:
        raise ValueError("No face detected for embedding")
    embedding = np.array(result[0]["embedding"])
    return embedding


def encrypt_embedding(embedding: np.ndarray) -> bytes:
    """Encrypt embedding array with Fernet symmetric encryption."""
    f = _get_fernet()
    data = embedding.tobytes()
    return f.encrypt(data)


def decrypt_embedding(encrypted: bytes, length: int = 512) -> np.ndarray:
    """Decrypt embedding bytes back to numpy array."""
    f = _get_fernet()
    data = f.decrypt(encrypted)
    arr = np.frombuffer(data, dtype=np.float64)
    return arr


def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine distance between two embeddings (0=identical, 1=different)."""
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    cosine_similarity = np.dot(emb1, emb2) / (norm1 * norm2)
    return float(1.0 - cosine_similarity)


def check_liveness(image_bytes: bytes) -> dict:
    """Run anti-spoofing liveness check."""
    try:
        from deepface import DeepFace
        img = image_bytes_to_array(image_bytes)
        faces = DeepFace.extract_faces(
            img_path=img,
            anti_spoofing=True,
            detector_backend=settings.detector_backend,
            enforce_detection=False,
        )
        if not faces:
            return {"is_real": False, "score": 0.0, "message": "No face detected"}
        face = faces[0]
        is_real = bool(face.get("is_real", False))
        score = float(face.get("antispoof_score", 0.0))
        return {"is_real": is_real, "score": score, "message": "Live" if is_real else "Spoof detected"}
    except Exception as e:
        log.warning("Liveness check error: %s", e)
        return {"is_real": False, "score": 0.0, "message": f"Error: {str(e)[:100]}"}
