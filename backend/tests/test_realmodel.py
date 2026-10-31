"""Real-model integration tests (MediaPipe + MiniFASNet + ArcFace on CPU).

Opt-in:  RUN_REAL_MODEL_TESTS=1 pytest backend/tests/test_realmodel.py -s
Requires models/face_landmarker.task, DeepFace weights in ~/.deepface and the
public test images in test_assets/ (LFW public-figure photos; not user data).

These verify that the real pipeline runs end to end and behaves sensibly on a
handful of images. Seven images say nothing statistically meaningful about
accuracy — use the evaluation module with a proper labelled dataset for that.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.config import settings
from backend.models.verification import OutcomeEnum
from backend.services.biometric.engine import Engine
from backend.services.biometric.templates import cosine_distance
from backend.services.policy_service import DEFAULTS, Policy

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "test_assets"
LANDMARKER = Path(settings.models_dir) / settings.landmarker_model_file

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_MODEL_TESTS") != "1" or not LANDMARKER.exists() or not ASSETS.exists(),
    reason="real-model tests are opt-in (RUN_REAL_MODEL_TESTS=1) and need model files + test_assets/",
)


@pytest.fixture(scope="module")
def provider():
    from backend.services.biometric.providers import DeepFaceProvider
    p = DeepFaceProvider(LANDMARKER)
    status = p.warmup()
    assert all(m["status"] == "READY" for m in status), status
    return p


def load(name: str) -> np.ndarray:
    img = cv2.imread(str(ASSETS / name))
    assert img is not None, name
    return img


def framed(name: str, face_width_px: int = 230) -> np.ndarray:
    """Place a tightly-cropped portrait on a 640x480 canvas, as a webcam would frame it."""
    img = load(name)
    s = face_width_px / img.shape[1]
    img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((480, 640, 3), 118, dtype=np.uint8)
    y0, x0 = (480 - img.shape[0]) // 2, (640 - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return canvas


def single_face(provider, img):
    a = provider.analyze(img)
    assert len(a.faces) == 1
    return a.faces[0]


def embed(provider, name):
    img = load(name)
    return provider.embed(img, single_face(provider, img))


def test_detection_counts(provider):
    for name in ("bush_valid_0.jpg", "powell_valid_0.jpg", "ref_bush_1.jpg"):
        assert len(provider.analyze(load(name)).faces) == 1, name
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    assert provider.analyze(blank).faces == []            # no whole-image fallback
    a, b = load("bush_valid_0.jpg"), load("powell_valid_0.jpg")
    h = min(a.shape[0], b.shape[0])
    collage = np.hstack([cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h)),
                         cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))])
    assert len(provider.analyze(collage).faces) == 2


def test_pose_and_blink_signals_are_plausible(provider):
    img = load("bush_valid_0.jpg")
    f = single_face(provider, img)
    flipped = single_face(provider, np.ascontiguousarray(cv2.flip(img, 1)))
    assert np.sign(f.yaw) == -np.sign(flipped.yaw)        # mirroring flips yaw sign
    assert 0.0 <= f.closure_any <= 1.0 and 0.0 <= f.sharpness <= 1.0


def test_embeddings_separate_identities(provider):
    bush = ["bush_valid_0.jpg", "bush_valid_1.jpg", "bush_valid_2.jpg", "ref_bush_1.jpg", "ref_bush_2.jpg"]
    powell = ["powell_valid_0.jpg", "ref_powell_1.jpg"]
    E = {n: embed(provider, n) for n in bush + powell}
    assert all(v.shape == (512,) for v in E.values())
    genuine = [cosine_distance(E[a], E[b]) for grp in (bush, powell) for a, b in combinations(grp, 2)]
    impostor = [cosine_distance(E[a], E[b]) for a in bush for b in powell]
    print(f"\n  genuine   n={len(genuine)} mean={np.mean(genuine):.3f} max={np.max(genuine):.3f}")
    print(f"  impostor  n={len(impostor)} mean={np.mean(impostor):.3f} min={np.min(impostor):.3f}")
    assert np.mean(genuine) < np.mean(impostor)
    assert np.max(genuine) < np.min(impostor), "this tiny public set is expected to be separable"


def test_bgr_input_matters(provider):
    img = load("bush_valid_0.jpg")
    f = single_face(provider, img)
    swapped = np.ascontiguousarray(img[:, :, ::-1])
    d = cosine_distance(provider.embed(img, f), provider.embed(swapped, f))
    print(f"\n  BGR vs channel-swapped embedding distance: {d:.3f}")
    assert d > 0.01


def test_passive_liveness_returns_labelled_confidence(provider):
    img = load("ref_bush_1.jpg")
    res = provider.passive_liveness(img, single_face(provider, img))
    print(f"\n  MiniFASNet on a scanned LFW photo: {res.label} ({res.confidence:.3f})")
    assert res.label in ("REAL", "SPOOF") and 0.0 <= res.confidence <= 1.0


class _Ops:
    def __init__(self, provider, img, ref):
        self.p, self.img, self.ref = provider, img, ref

    def passive(self, f):
        return self.p.passive_liveness(self.img, f)

    def embed(self, f):
        return self.p.embed(self.img, f)

    def distance(self, v):
        return cosine_distance(v, self.ref)


def _run(provider, img, ref, policy):
    eng = Engine(policy, "VERIFY")
    st, _ = eng.start(datetime.utcnow())
    now = datetime.utcnow()
    analysis = provider.analyze(img)
    d = None
    for _ in range(30):
        now += timedelta(milliseconds=400)
        d = eng.process(st, analysis, now, _Ops(provider, img, ref))
        if d.terminal:
            break
    return d


def test_tight_portrait_is_rejected_by_framing_guidance(provider):
    from backend.services.biometric.analysis import assess_quality
    q = assess_quality(single_face(provider, load("bush_valid_0.jpg")), 0.20, 0.75, True, 0.40)
    assert "FACE_TOO_LARGE" in q.issues


def test_real_pipeline_match_and_non_match(provider):
    ref_img = framed("ref_bush_1.jpg")
    ref = provider.embed(ref_img, single_face(provider, ref_img))
    pol = Policy(1, {**DEFAULTS, "active_liveness_required": False, "passive_liveness_required": False})
    results = {}
    for probe in ("bush_valid_0.jpg", "bush_valid_1.jpg", "bush_valid_2.jpg", "ref_bush_2.jpg"):
        d = _run(provider, framed(probe), ref, pol)
        results[probe] = (d.outcome.value, d.reason, None if d.distance is None else round(d.distance, 3))
    impostor = _run(provider, framed("ref_powell_1.jpg"), ref, pol)
    print("\n  genuine probes:", results)
    print(f"  impostor → {impostor.outcome.value} d={impostor.distance:.3f}")
    assert impostor.outcome == OutcomeEnum.NOT_MATCHED
    decided = [r for r in results.values() if r[0] in ("VERIFIED", "NOT_MATCHED")]
    assert any(r[0] == "VERIFIED" for r in decided), results


def test_still_photo_cannot_pass_active_challenge(provider):
    ref_img = framed("ref_bush_1.jpg")
    ref = provider.embed(ref_img, single_face(provider, ref_img))
    pol = Policy(1, {**DEFAULTS, "passive_liveness_required": False})
    d = _run(provider, framed("bush_valid_0.jpg"), ref, pol)
    assert d.outcome != OutcomeEnum.VERIFIED
