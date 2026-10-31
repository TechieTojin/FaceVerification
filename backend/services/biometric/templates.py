"""Encrypted template storage and embedding comparison.

Template plaintext format v2::

    b"FVT2" | uint16 header_len | header JSON | float32 matrix bytes (row-major)

Header: {"model": str, "dim": int, "rows": int, "dtype": "float32", "norm": "l2"}.
Vectors are L2-normalised before storage, so cosine distance is 1 - dot.
Anything that does not carry the FVT2 tag is a legacy v2.1 template, which was
produced by a different (RGB-swapped, Haar-detected) pipeline and is refused.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

from backend.config import settings

_MAGIC = b"FVT2"
_fernet: Optional[Fernet] = None


class TemplateError(Exception):
    """Base class; message is safe to show to operators."""


class LegacyTemplateError(TemplateError):
    pass


class TemplateMismatchError(TemplateError):
    pass


class TemplateDecryptError(TemplateError):
    pass


@dataclass(frozen=True)
class DecodedTemplate:
    model: str
    vectors: np.ndarray  # shape (rows, dim), float32, L2-normalised

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def vector(self) -> np.ndarray:
        return self.vectors[0]


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        _fernet = Fernet(settings.get_fernet_key())
    return _fernet


def reset_crypto_cache() -> None:
    global _fernet
    _fernet = None


def l2_normalise(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise TemplateMismatchError("Embedding contains invalid values")
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise TemplateMismatchError("Embedding has zero magnitude")
    return (v / n).astype(np.float32)


def encrypt_vectors(vectors: list[np.ndarray] | np.ndarray, model: str) -> bytes:
    mat = np.vstack([l2_normalise(v) for v in vectors]).astype(np.float32)
    header = json.dumps({"model": model, "dim": int(mat.shape[1]), "rows": int(mat.shape[0]),
                         "dtype": "float32", "norm": "l2"}).encode()
    plaintext = _MAGIC + struct.pack(">H", len(header)) + header + mat.tobytes()
    return _get_fernet().encrypt(plaintext)


def decrypt_vectors(blob: bytes) -> DecodedTemplate:
    try:
        plaintext = _get_fernet().decrypt(blob)
    except InvalidToken as exc:
        raise TemplateDecryptError("Stored template could not be decrypted with the configured key") from exc
    if not plaintext.startswith(_MAGIC):
        raise LegacyTemplateError("Template was created by v2.x and must be re-enrolled")
    (hlen,) = struct.unpack(">H", plaintext[4:6])
    header = json.loads(plaintext[6:6 + hlen])
    body = plaintext[6 + hlen:]
    dim, rows = int(header["dim"]), int(header["rows"])
    if len(body) != dim * rows * 4:
        raise TemplateMismatchError("Template payload is corrupt")
    mat = np.frombuffer(body, dtype=np.float32).reshape(rows, dim).copy()
    return DecodedTemplate(model=str(header["model"]), vectors=mat)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2]. Refuses dimension mismatch instead of truncating."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.shape != b.shape:
        raise TemplateMismatchError(f"Embedding dimension mismatch ({a.size} vs {b.size})")
    an, bn = l2_normalise(a), l2_normalise(b)
    return float(np.clip(1.0 - float(np.dot(an, bn)), 0.0, 2.0))


def compare_to_template(live: np.ndarray, template: DecodedTemplate, live_model: str) -> float:
    if template.model != live_model:
        raise TemplateMismatchError(
            f"Reference template was produced by {template.model}, live embedding by {live_model}")
    return cosine_distance(live, template.vector)


@dataclass(frozen=True)
class BuiltTemplate:
    vector: np.ndarray
    consistency: float      # max pairwise distance between samples
    sample_count: int


def build_template(samples: list[np.ndarray]) -> BuiltTemplate:
    """Average L2-normalised samples into one reference vector."""
    if len(samples) < 1:
        raise TemplateMismatchError("No samples")
    normed = [l2_normalise(s) for s in samples]
    dims = {v.shape[0] for v in normed}
    if len(dims) != 1:
        raise TemplateMismatchError("Samples have different dimensions")
    worst = max((cosine_distance(a, b) for a, b in combinations(normed, 2)), default=0.0)
    mean = l2_normalise(np.mean(np.vstack(normed), axis=0))
    return BuiltTemplate(vector=mean, consistency=worst, sample_count=len(normed))
