"""
Unit tests for face_verification.py (v1.1)

Run with:
    .venv/Scripts/pytest tests/ -v
"""

import argparse
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure the project root is on sys.path so we can import face_verification
sys.path.insert(0, str(Path(__file__).parent.parent))

import face_verification as fv


# ── helpers ────────────────────────────────────────────────────────────────────

def _blank_frame(h: int = 100, w: int = 100) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _dummy_ref() -> np.ndarray:
    """A non-blank reference image (white square)."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    return img


# ── load_reference_image ───────────────────────────────────────────────────────

class TestLoadReferenceImage:
    def test_exits_when_path_empty(self):
        with pytest.raises(SystemExit) as exc:
            fv.load_reference_image("")
        assert exc.value.code == 1

    def test_exits_when_file_missing(self, tmp_path):
        missing = str(tmp_path / "nonexistent.jpg")
        with pytest.raises(SystemExit) as exc:
            fv.load_reference_image(missing)
        assert exc.value.code == 1

    def test_exits_when_file_unreadable(self, tmp_path):
        # Write a non-image file
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image")
        with pytest.raises(SystemExit) as exc:
            fv.load_reference_image(str(bad))
        assert exc.value.code == 1

    def test_loads_valid_image(self, tmp_path):
        import cv2
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        path = str(tmp_path / "ref.jpg")
        cv2.imwrite(path, img)
        loaded = fv.load_reference_image(path)
        assert loaded is not None
        assert loaded.shape == (50, 50, 3)


# ── open_camera ────────────────────────────────────────────────────────────────

class TestOpenCamera:
    def test_exits_on_camera_failure(self):
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap
            with pytest.raises(SystemExit) as exc:
                fv.open_camera(0)
            assert exc.value.code == 1

    def test_returns_capture_on_success(self):
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 640.0
            mock_cap_cls.return_value = mock_cap
            result = fv.open_camera(0)
            assert result is mock_cap

    def test_configures_resolution(self):
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 640.0
            mock_cap_cls.return_value = mock_cap
            fv.open_camera(0)
            # Verify CAP_PROP_FRAME_WIDTH was set
            set_calls = [str(c) for c in mock_cap.set.call_args_list]
            assert any("3" in s for s in set_calls)  # CAP_PROP_FRAME_WIDTH == 3


# ── verify_face ────────────────────────────────────────────────────────────────

class TestVerifyFace:
    def _mock_deepface(self, verified: bool):
        mock_df = MagicMock()
        mock_df.verify.return_value = {"verified": verified}
        return mock_df

    def test_returns_true_on_match(self):
        # DeepFace is imported lazily inside verify_face; patch the module in sys.modules
        mock_deepface_module = types.ModuleType("deepface")
        mock_df = MagicMock()
        mock_df.verify.return_value = {"verified": True}
        mock_deepface_module.DeepFace = mock_df
        with patch.dict("sys.modules", {"deepface": mock_deepface_module}):
            result = fv.verify_face(_blank_frame(), _dummy_ref(), "ArcFace", "opencv")
        assert result is True

    def test_returns_false_on_value_error(self):
        mock_deepface_module = types.ModuleType("deepface")
        mock_df = MagicMock()
        mock_df.verify.side_effect = ValueError("Face could not be detected")
        mock_deepface_module.DeepFace = mock_df
        with patch.dict("sys.modules", {"deepface": mock_deepface_module}):
            result = fv.verify_face(_blank_frame(), _dummy_ref(), "ArcFace", "opencv")
        assert result is False

    def test_returns_none_on_unexpected_error(self):
        mock_deepface_module = types.ModuleType("deepface")
        mock_df = MagicMock()
        mock_df.verify.side_effect = RuntimeError("GPU out of memory")
        mock_deepface_module.DeepFace = mock_df
        with patch.dict("sys.modules", {"deepface": mock_deepface_module}):
            result = fv.verify_face(_blank_frame(), _dummy_ref(), "ArcFace", "opencv")
        assert result is None

    def test_does_not_mutate_reference(self):
        mock_deepface_module = types.ModuleType("deepface")
        mock_df = MagicMock()
        mock_df.verify.return_value = {"verified": False}
        mock_deepface_module.DeepFace = mock_df
        ref = _dummy_ref()
        ref_copy = ref.copy()
        with patch.dict("sys.modules", {"deepface": mock_deepface_module}):
            fv.verify_face(_blank_frame(), ref, "ArcFace", "opencv")
        np.testing.assert_array_equal(ref, ref_copy)


# ── load_config ────────────────────────────────────────────────────────────────

class TestLoadConfig:
    def _make_args(self, reference=None, camera=None):
        return argparse.Namespace(reference=reference, camera=camera)

    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("REFERENCE_IMAGE_PATH", "/some/path.jpg")
        monkeypatch.setenv("CAMERA_INDEX", "2")
        monkeypatch.setenv("FACE_MODEL", "FaceNet")
        monkeypatch.setenv("DETECTOR_BACKEND", "ssd")
        cfg = fv.load_config(self._make_args())
        assert cfg["reference_path"] == "/some/path.jpg"
        assert cfg["camera_index"] == 2
        assert cfg["model"] == "FaceNet"
        assert cfg["detector"] == "ssd"

    def test_cli_args_override_env(self, monkeypatch):
        monkeypatch.setenv("REFERENCE_IMAGE_PATH", "/env/path.jpg")
        monkeypatch.setenv("CAMERA_INDEX", "1")
        cfg = fv.load_config(self._make_args(reference="/cli/path.jpg", camera=3))
        assert cfg["reference_path"] == "/cli/path.jpg"
        assert cfg["camera_index"] == 3

    def test_defaults_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("REFERENCE_IMAGE_PATH", raising=False)
        monkeypatch.delenv("CAMERA_INDEX", raising=False)
        monkeypatch.delenv("FACE_MODEL", raising=False)
        monkeypatch.delenv("DETECTOR_BACKEND", raising=False)
        cfg = fv.load_config(self._make_args())
        assert cfg["reference_path"] == ""
        assert cfg["camera_index"] == 0
        assert cfg["model"] == "ArcFace"
        assert cfg["detector"] == "opencv"


# ── draw_overlay ───────────────────────────────────────────────────────────────

class TestDrawOverlay:
    def test_does_not_raise_on_match(self):
        frame = _blank_frame(480, 640)
        fv.draw_overlay(frame, True, 30.0, 250.0)

    def test_does_not_raise_on_no_match(self):
        frame = _blank_frame(480, 640)
        fv.draw_overlay(frame, False, 30.0, 250.0)

    def test_does_not_raise_on_none(self):
        frame = _blank_frame(480, 640)
        fv.draw_overlay(frame, None, 0.0, 0.0)

    def test_match_writes_pixels(self):
        frame = _blank_frame(480, 640)
        fv.draw_overlay(frame, True, 30.0, 100.0)
        # Frame must be non-zero (text was drawn)
        assert frame.sum() > 0

    def test_no_match_writes_pixels(self):
        frame = _blank_frame(480, 640)
        fv.draw_overlay(frame, False, 30.0, 100.0)
        assert frame.sum() > 0
