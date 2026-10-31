"""Download model files that are not bundled with Python packages.

    python scripts/fetch_models.py

* models/face_landmarker.task — MediaPipe FaceLandmarker (server-side landmarks)
* frontend/public/models/face_landmarker.task — same file for in-browser tracking

DeepFace downloads ArcFace / MiniFASNet weights into ~/.deepface on first use.
Model files are gitignored; never commit them.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    dest_dir = ROOT / "models"
    dest_dir.mkdir(exist_ok=True)
    for name, (url, digest) in MODELS.items():
        dest = dest_dir / name
        if not dest.exists() or sha256(dest) != digest:
            print(f"Downloading {name} …")
            tmp = dest.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 — fixed https URL
            got = sha256(tmp)
            if got != digest:
                tmp.unlink(missing_ok=True)
                print(f"Checksum mismatch for {name}: {got}", file=sys.stderr)
                return 1
            tmp.replace(dest)
        print(f"OK  {dest.relative_to(ROOT)}")
        web = ROOT / "frontend" / "public" / "models" / name
        web.parent.mkdir(parents=True, exist_ok=True)
        if not web.exists() or sha256(web) != digest:
            shutil.copyfile(dest, web)
        print(f"OK  {web.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
