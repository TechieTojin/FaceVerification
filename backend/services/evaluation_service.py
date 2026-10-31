"""Biometric evaluation on administrator-supplied labelled data.

FAR/FRR/EER are computed *only* from a labelled dataset uploaded for the run
(ZIP of `<identity>/<image>` files). Production verification outcomes are never
used: they have no ground truth. Images and embeddings are processed in memory
and discarded; only aggregate distances/curves are stored.

Definitions at threshold t (distance ≤ t ⇒ accept):
  FAR(t) = impostor pairs accepted / impostor pairs
  FRR(t) = genuine pairs rejected  / genuine pairs
  EER    = the rate where FAR and FRR cross (linear interpolation on a fine grid)
"""
from __future__ import annotations

import io
import json
import logging
import random
import threading
import zipfile
from datetime import datetime
from itertools import combinations
from typing import Any, Optional

import numpy as np

from backend.database import SessionLocal
from backend.models.documents import EvaluationRun
from backend.services.biometric.imaging import InvalidImageError, decode_image
from backend.services.biometric.templates import cosine_distance
from backend.services.inference import runtime

log = logging.getLogger(__name__)

MAX_ZIP_BYTES = 50 * 1024 * 1024
MAX_IMAGES = 500
MAX_IMPOSTOR_PAIRS = 50_000
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")


class EvaluationInputError(ValueError):
    pass


def compute_metrics(genuine: list[float], impostor: list[float], policy_threshold: float) -> dict[str, Any]:
    if not genuine or not impostor:
        raise EvaluationInputError("Need at least one genuine and one impostor pair")
    g = np.sort(np.asarray(genuine, dtype=np.float64))
    i = np.sort(np.asarray(impostor, dtype=np.float64))
    grid = np.round(np.arange(0.0, 1.2001, 0.005), 4)
    far = np.searchsorted(i, grid, side="right") / i.size
    frr = 1.0 - np.searchsorted(g, grid, side="right") / g.size
    diff = far - frr
    k = int(np.argmax(diff >= 0)) if np.any(diff >= 0) else len(grid) - 1
    if k > 0 and diff[k] != diff[k - 1]:
        w = -diff[k - 1] / (diff[k] - diff[k - 1])
        eer_t = float(grid[k - 1] + w * (grid[k] - grid[k - 1]))
        eer = float(far[k - 1] + w * (far[k] - far[k - 1]))
    else:
        eer_t, eer = float(grid[k]), float((far[k] + frr[k]) / 2)

    def at(t: float) -> dict[str, float]:
        return {"threshold": round(t, 4),
                "far": round(float(np.searchsorted(i, t, side="right") / i.size), 6),
                "frr": round(float(1.0 - np.searchsorted(g, t, side="right") / g.size), 6)}

    curve = [{"threshold": float(t), "far": round(float(a), 6), "frr": round(float(b), 6)}
             for t, a, b in zip(grid[::4], far[::4], frr[::4])]
    warnings = []
    if g.size < 100:
        warnings.append(f"Only {g.size} genuine pairs — FRR estimates are statistically unreliable.")
    if i.size < 1000:
        warnings.append(f"Only {i.size} impostor pairs — FAR below {1 / max(i.size, 1):.4f} cannot be resolved.")
    return {
        "genuine_pairs": int(g.size), "impostor_pairs": int(i.size),
        "eer": round(eer, 6), "eer_threshold": round(eer_t, 4),
        "at_policy_threshold": at(policy_threshold),
        "operating_points": [at(t) for t in (0.3, 0.4, 0.5, 0.6, 0.68)],
        "genuine_distance": {"mean": round(float(g.mean()), 4), "p95": round(float(np.percentile(g, 95)), 4),
                             "max": round(float(g.max()), 4)},
        "impostor_distance": {"mean": round(float(i.mean()), 4), "p5": round(float(np.percentile(i, 5)), 4),
                              "min": round(float(i.min()), 4)},
        "curve": curve,
        "warnings": warnings,
    }


def parse_dataset(data: bytes) -> dict[str, list[bytes]]:
    if len(data) > MAX_ZIP_BYTES:
        raise EvaluationInputError("Dataset exceeds 50 MB")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise EvaluationInputError("Upload a ZIP archive") from exc
    groups: dict[str, list[bytes]] = {}
    count = 0
    for info in zf.infolist():
        if info.is_dir() or not info.filename.lower().endswith(IMAGE_EXT):
            continue
        parts = [p for p in info.filename.replace("\\", "/").split("/") if p and not p.startswith("__MACOSX")]
        if len(parts) < 2:
            continue
        if info.file_size > 10 * 1024 * 1024:
            raise EvaluationInputError(f"{parts[-1]} is larger than 10 MB")
        count += 1
        if count > MAX_IMAGES:
            raise EvaluationInputError(f"Dataset has more than {MAX_IMAGES} images")
        groups.setdefault(parts[-2], []).append(zf.read(info))
    usable = {k: v for k, v in groups.items() if v}
    if len(usable) < 2:
        raise EvaluationInputError("Need at least two identity folders")
    if not any(len(v) >= 2 for v in usable.values()):
        raise EvaluationInputError("At least one identity needs two or more images for genuine pairs")
    return usable


def _embed_one(provider, data: bytes) -> Optional[np.ndarray]:
    try:
        bgr = decode_image(data, max_side=1280)
    except InvalidImageError:
        return None
    analysis = provider.analyze(bgr)
    if len(analysis.faces) != 1:
        return None
    return provider.embed(bgr, analysis.faces[0])


def run_evaluation(run_id: str, groups: dict[str, list[bytes]], policy_threshold: float) -> None:
    db = SessionLocal()
    try:
        run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        embeddings: dict[str, list[np.ndarray]] = {}
        skipped = 0
        for ident, images in groups.items():
            for img in images:
                vec = runtime.run(_embed_one, runtime.provider, img, timeout=60.0)
                if vec is None:
                    skipped += 1
                else:
                    embeddings.setdefault(ident, []).append(vec)
        genuine = [cosine_distance(a, b) for vecs in embeddings.values() for a, b in combinations(vecs, 2)]
        idents = list(embeddings)
        impostor_pairs = [(a, b) for x, y in combinations(idents, 2) for a in embeddings[x] for b in embeddings[y]]
        if len(impostor_pairs) > MAX_IMPOSTOR_PAIRS:
            impostor_pairs = random.Random(7).sample(impostor_pairs, MAX_IMPOSTOR_PAIRS)
        impostor = [cosine_distance(a, b) for a, b in impostor_pairs]
        run.identity_count = len(embeddings)
        run.image_count = sum(len(v) for v in groups.values())
        run.skipped_images = skipped
        results = compute_metrics(genuine, impostor, policy_threshold)
        run.genuine_pairs, run.impostor_pairs = results["genuine_pairs"], results["impostor_pairs"]
        run.results_json = json.dumps(results)
        run.status = "COMPLETED"
    except Exception as exc:  # noqa: BLE001
        log.exception("Evaluation %s failed", run_id)
        run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        if run:
            run.status = "FAILED"
            run.error = str(exc)[:500] if isinstance(exc, EvaluationInputError) else f"{type(exc).__name__}: evaluation failed"
    finally:
        if run:
            run.completed_at = datetime.utcnow()
            db.commit()
        db.close()


def start_evaluation(run_id: str, groups: dict[str, list[bytes]], policy_threshold: float) -> None:
    threading.Thread(target=run_evaluation, args=(run_id, groups, policy_threshold), daemon=True,
                     name=f"evaluation-{run_id[:8]}").start()
