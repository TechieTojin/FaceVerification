"""Tests for AppConfig loading from environment variables."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.config import AppConfig


class TestConfigFromEnv:
    def test_defaults_when_no_env_vars(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove all relevant keys
            keys = [
                "REFERENCE_IMAGE_PATH", "CAMERA_INDEX", "CAMERA_WIDTH", "CAMERA_HEIGHT",
                "RECOGNITION_MODEL", "DETECTOR_BACKEND", "ENABLE_LIVENESS",
                "ENABLE_PERFORMANCE_OVERLAY", "INFERENCE_INTERVAL_MS", "RESULT_EXPIRY_MS",
                "MAX_VERIFICATION_ATTEMPTS", "LOG_LEVEL",
            ]
            env = {k: v for k, v in os.environ.items() if k not in keys}
            with patch.dict(os.environ, env, clear=True):
                cfg = AppConfig.from_env()
        assert cfg.reference_image_path == ""
        assert cfg.camera_index == 0
        assert cfg.recognition_model == "ArcFace"
        assert cfg.detector_backend == "opencv"
        assert cfg.enable_liveness is True
        assert cfg.enable_performance_overlay is True
        assert cfg.inference_interval_ms == 500.0
        assert cfg.result_expiry_ms == 1500.0
        assert cfg.log_level == "INFO"

    def test_loads_reference_image_path(self):
        with patch.dict(os.environ, {"REFERENCE_IMAGE_PATH": "/tmp/ref.jpg"}):
            cfg = AppConfig.from_env()
        assert cfg.reference_image_path == "/tmp/ref.jpg"

    def test_missing_reference_image_path_is_empty_string(self):
        env = {k: v for k, v in os.environ.items() if k != "REFERENCE_IMAGE_PATH"}
        with patch.dict(os.environ, env, clear=True):
            cfg = AppConfig.from_env()
        assert cfg.reference_image_path == ""

    def test_invalid_camera_index_defaults_to_zero(self):
        with patch.dict(os.environ, {"CAMERA_INDEX": "not_a_number"}):
            cfg = AppConfig.from_env()
        assert cfg.camera_index == 0

    def test_valid_camera_index(self):
        with patch.dict(os.environ, {"CAMERA_INDEX": "2"}):
            cfg = AppConfig.from_env()
        assert cfg.camera_index == 2

    def test_enable_liveness_false(self):
        with patch.dict(os.environ, {"ENABLE_LIVENESS": "false"}):
            cfg = AppConfig.from_env()
        assert cfg.enable_liveness is False

    def test_enable_liveness_true(self):
        with patch.dict(os.environ, {"ENABLE_LIVENESS": "true"}):
            cfg = AppConfig.from_env()
        assert cfg.enable_liveness is True

    def test_enable_liveness_zero(self):
        with patch.dict(os.environ, {"ENABLE_LIVENESS": "0"}):
            cfg = AppConfig.from_env()
        assert cfg.enable_liveness is False

    def test_recognition_model_loaded(self):
        with patch.dict(os.environ, {"RECOGNITION_MODEL": "VGG-Face"}):
            cfg = AppConfig.from_env()
        assert cfg.recognition_model == "VGG-Face"

    def test_log_level_loaded(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            cfg = AppConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_result_expiry_ms_loaded(self):
        with patch.dict(os.environ, {"RESULT_EXPIRY_MS": "2000"}):
            cfg = AppConfig.from_env()
        assert cfg.result_expiry_ms == 2000.0

    def test_invalid_result_expiry_defaults(self):
        with patch.dict(os.environ, {"RESULT_EXPIRY_MS": "bad"}):
            cfg = AppConfig.from_env()
        assert cfg.result_expiry_ms == 1500.0


class TestCliOverrides:
    def test_reference_override(self):
        cfg = AppConfig()
        cfg.apply_cli_overrides(reference="/tmp/other.jpg", camera=None)
        assert cfg.reference_image_path == "/tmp/other.jpg"

    def test_camera_override(self):
        cfg = AppConfig()
        cfg.apply_cli_overrides(reference=None, camera=3)
        assert cfg.camera_index == 3

    def test_none_overrides_ignored(self):
        cfg = AppConfig(reference_image_path="/orig.jpg", camera_index=1)
        cfg.apply_cli_overrides(reference=None, camera=None)
        assert cfg.reference_image_path == "/orig.jpg"
        assert cfg.camera_index == 1
