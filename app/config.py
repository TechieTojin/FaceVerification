"""Dataclass-based application configuration loaded from environment / dotenv."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class AppConfig:
    reference_image_path: str = ""
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    recognition_model: str = "ArcFace"
    detector_backend: str = "opencv"
    enable_liveness: bool = True
    enable_performance_overlay: bool = True
    inference_interval_ms: float = 500.0
    result_expiry_ms: float = 1500.0
    max_verification_attempts: int = 10
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build config from environment variables (dotenv should be loaded first)."""
        def _int(key: str, default: int) -> int:
            raw = os.getenv(key, "")
            try:
                return int(raw)
            except (ValueError, TypeError):
                if raw:
                    log.warning("Invalid %s=%r — using default %d", key, raw, default)
                return default

        def _float(key: str, default: float) -> float:
            raw = os.getenv(key, "")
            try:
                return float(raw)
            except (ValueError, TypeError):
                if raw:
                    log.warning("Invalid %s=%r — using default %f", key, raw, default)
                return default

        def _bool(key: str, default: bool) -> bool:
            raw = os.getenv(key, "").strip().lower()
            if raw in ("1", "true", "yes", "on"):
                return True
            if raw in ("0", "false", "no", "off"):
                return False
            return default

        return cls(
            reference_image_path=os.getenv("REFERENCE_IMAGE_PATH", ""),
            camera_index=_int("CAMERA_INDEX", 0),
            camera_width=_int("CAMERA_WIDTH", 640),
            camera_height=_int("CAMERA_HEIGHT", 480),
            recognition_model=os.getenv("RECOGNITION_MODEL", "ArcFace"),
            detector_backend=os.getenv("DETECTOR_BACKEND", "opencv"),
            enable_liveness=_bool("ENABLE_LIVENESS", True),
            enable_performance_overlay=_bool("ENABLE_PERFORMANCE_OVERLAY", True),
            inference_interval_ms=_float("INFERENCE_INTERVAL_MS", 500.0),
            result_expiry_ms=_float("RESULT_EXPIRY_MS", 1500.0),
            max_verification_attempts=_int("MAX_VERIFICATION_ATTEMPTS", 10),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def apply_cli_overrides(self, reference: str | None, camera: int | None) -> None:
        """Override config fields from CLI arguments."""
        if reference is not None:
            self.reference_image_path = reference
        if camera is not None:
            self.camera_index = camera
