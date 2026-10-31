"""Seed a REAL ArcFace template for the real-model browser tests.

Test-only: refuses to run unless E2E_REAL_SEED=1. Reads the throwaway per-run
settings written by frontend/e2e/start-backend.mjs (.e2e/backend-env-real.json).

    E2E_REAL_SEED=1 python -m backend.tests.real_seed <email> <image-in-test_assets>
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str]) -> int:
    if os.environ.get("E2E_REAL_SEED") != "1" or len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    os.environ.update(json.loads((ROOT / ".e2e" / "backend-env-real.json").read_text()))
    import cv2
    from backend.config import settings
    from backend.database import SessionLocal
    from backend.models.biometric import BiometricEnrolment, EnrolmentStatus
    from backend.models.user import User
    from backend.services.biometric.providers import DeepFaceProvider
    from backend.services.biometric.templates import encrypt_vectors
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_test_videos import framed  # same framing as the browser clips

    img = framed(cv2.imread(str(ROOT / "test_assets" / argv[2])))
    provider = DeepFaceProvider(settings.landmarker_model_path)
    faces = provider.analyze(img).faces
    if len(faces) != 1:
        print("expected exactly one face", file=sys.stderr)
        return 1
    vec = provider.embed(img, faces[0])
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == argv[1]).one()
        enr = db.query(BiometricEnrolment).filter(BiometricEnrolment.user_id == user.id).first()
        if enr is None:
            enr = BiometricEnrolment(user_id=user.id)
            db.add(enr)
        enr.embedding_encrypted = encrypt_vectors([vec], provider.recognition_model)
        enr.status, enr.template_version, enr.embedding_dim = EnrolmentStatus.ACTIVE, 2, len(vec)
        enr.consent_given, enr.consent_timestamp, enr.enrolled_at = True, datetime.utcnow(), datetime.utcnow()
        enr.sample_count, enr.quality_score = 1, 0.8
        db.commit()
    finally:
        db.close()
    print(f"seeded real template for {argv[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
