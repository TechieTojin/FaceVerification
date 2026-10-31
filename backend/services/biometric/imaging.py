"""Safe image decoding for untrusted uploads."""
from __future__ import annotations

import io
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_PIXELS = 12_000_000      # ~4000x3000; rejects decompression bombs well before PIL's limit
MAX_DIMENSION = 6000


class InvalidImageError(ValueError):
    """Raised for unreadable, unsupported or oversized images (message safe for clients)."""


def decode_image(data: bytes, max_side: int | None = None) -> np.ndarray:
    """Decode JPEG/PNG/WebP bytes into a BGR uint8 array.

    Applies EXIF orientation, enforces format and size limits and optionally
    downscales so the longest side is at most `max_side`.
    """
    if not data:
        raise InvalidImageError("Empty image")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
            fmt = img.format
            if fmt not in ALLOWED_FORMATS:
                raise InvalidImageError(f"Unsupported image format ({fmt or 'unknown'}); use JPEG, PNG or WebP")
            w, h = img.size
            if w * h > MAX_PIXELS or max(w, h) > MAX_DIMENSION:
                raise InvalidImageError("Image dimensions are too large")
            if min(w, h) < 64:
                raise InvalidImageError("Image is too small")
            img.load()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise InvalidImageError("Image could not be decoded") from exc
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    rgb = np.asarray(img, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
