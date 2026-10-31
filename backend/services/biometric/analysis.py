"""Provider-neutral frame analysis types and face-crop quality scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FaceObservation:
    """One detected face. Normalised coordinates are fractions of frame size."""

    bbox: tuple[float, float, float, float]          # x, y, w, h (normalised)
    yaw: float                                       # +ve = subject turned to *their* left
    pitch: float
    roll: float
    blink_left: float                                # 0 open … 1 closed
    blink_right: float
    sharpness: float                                 # 0..1 normalised Laplacian variance of crop
    brightness: float                                # 0..1 mean luminance of crop
    detection_score: float = 1.0
    # Provider-private pixel geometry used for cropping/alignment.
    bbox_px: tuple[int, int, int, int] = (0, 0, 0, 0)
    eye_left_px: tuple[float, float] = (0.0, 0.0)
    eye_right_px: tuple[float, float] = (0.0, 0.0)
    extra: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2, y + h / 2

    @property
    def closure_both(self) -> float:
        """Eye closure that *both* eyes reach (for a deliberate blink)."""
        return min(self.blink_left, self.blink_right)

    @property
    def closure_any(self) -> float:
        """Largest closure of either eye (both eyes open ⇔ this is low)."""
        return max(self.blink_left, self.blink_right)


@dataclass
class FrameAnalysis:
    width: int
    height: int
    faces: list[FaceObservation]
    analyze_ms: float = 0.0


@dataclass(frozen=True)
class PassiveResult:
    label: str            # "REAL" | "SPOOF"
    confidence: float     # confidence of *that* label (not P(real))
    ms: float = 0.0


@dataclass
class QualityReport:
    score: float
    components: dict[str, float]
    issues: list[str]

    @property
    def guidance(self) -> list[str]:
        return [GUIDANCE[i] for i in self.issues if i in GUIDANCE]


GUIDANCE = {
    "NO_FACE": "No face detected — look at the camera",
    "MULTIPLE_FACES": "Only one person should be visible",
    "FACE_TOO_SMALL": "Move closer",
    "FACE_TOO_LARGE": "Move back slightly",
    "OUT_OF_FRAME": "Center your face in the frame",
    "TOO_DARK": "Improve lighting",
    "TOO_BRIGHT": "Reduce bright light or glare",
    "BLURRY": "Hold still",
    "POSE": "Look directly at the camera",
    "LOW_QUALITY": "Improve lighting and hold still",
}


def assess_quality(face: FaceObservation, min_width_ratio: float, max_width_ratio: float,
                   require_frontal: bool, min_score: Optional[float] = None) -> QualityReport:
    """Composite quality of a face crop with actionable issue codes.

    score = 0.35·sharpness + 0.25·exposure + 0.20·size + 0.20·pose   (each in 0..1)
    """
    x, y, w, h = face.bbox
    issues: list[str] = []

    if w < min_width_ratio:
        issues.append("FACE_TOO_SMALL")
    elif w > max_width_ratio:
        issues.append("FACE_TOO_LARGE")
    margin = 0.01
    if x < margin or y < margin or x + w > 1 - margin or y + h > 1 - margin:
        issues.append("OUT_OF_FRAME")
    if face.brightness < 0.22:
        issues.append("TOO_DARK")
    elif face.brightness > 0.85:
        issues.append("TOO_BRIGHT")
    if face.sharpness < 0.25:
        issues.append("BLURRY")
    if require_frontal and (abs(face.yaw) > 12 or abs(face.pitch) > 18):
        issues.append("POSE")

    sharp_s = min(1.0, face.sharpness)
    expo_s = max(0.0, 1.0 - abs(face.brightness - 0.5) * 2.2)
    size_s = min(1.0, w / max(min_width_ratio * 1.5, 1e-6))
    pose_s = max(0.0, 1.0 - max(abs(face.yaw), abs(face.pitch)) / 30.0)
    score = round(0.35 * sharp_s + 0.25 * expo_s + 0.20 * size_s + 0.20 * pose_s, 4)
    if min_score is not None and score < min_score and not issues:
        issues.append("LOW_QUALITY")
    return QualityReport(score=score, components={
        "sharpness": round(sharp_s, 3), "exposure": round(expo_s, 3),
        "size": round(size_s, 3), "pose": round(pose_s, 3),
    }, issues=issues)
