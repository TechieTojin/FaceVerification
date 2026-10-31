"""Create Y4M test videos for Chromium's fake camera (real-model browser tests).

    python scripts/make_test_videos.py

Reads public LFW images from test_assets/ (gitignored, not user data) and
writes static 640x480 clips to .e2e/videos/. A static clip is, by design, a
"photo held up to the camera": it can never complete an active liveness
challenge, which the real-model tests rely on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "test_assets"
OUT = ROOT / ".e2e" / "videos"
CLIPS = {"genuine-a": "bush_valid_2.jpg", "genuine-b": "bush_valid_1.jpg", "impostor": "ref_powell_1.jpg"}


def framed(img: np.ndarray, face_width_px: int = 230) -> np.ndarray:
    s = face_width_px / img.shape[1]
    img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((480, 640, 3), 118, dtype=np.uint8)
    y0, x0 = (480 - img.shape[0]) // 2, (640 - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return canvas


def write_y4m(path: Path, frame_bgr: np.ndarray, frames: int = 150) -> None:
    """Static scene with mild per-frame sensor noise, like a real webcam pointed at a photo."""
    rng = np.random.default_rng(42)
    with path.open("wb") as f:
        f.write(b"YUV4MPEG2 W640 H480 F30:1 Ip A1:1 C420jpeg\n")
        for _ in range(frames):
            noisy = np.clip(frame_bgr.astype(np.int16) + rng.normal(0, 2.0, frame_bgr.shape), 0, 255).astype(np.uint8)
            f.write(b"FRAME\n")
            f.write(cv2.cvtColor(noisy, cv2.COLOR_BGR2YUV_I420).tobytes())


def main() -> int:
    if not ASSETS.exists():
        print("test_assets/ not found — real-model browser tests will be skipped", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for name, file in CLIPS.items():
        img = cv2.imread(str(ASSETS / file))
        if img is None:
            print(f"missing {file}", file=sys.stderr)
            return 1
        write_y4m(OUT / f"{name}.y4m", framed(img))
        print(f"wrote {OUT / f'{name}.y4m'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
