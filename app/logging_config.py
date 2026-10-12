"""Logging configuration for FaceVerification."""
from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a standard format."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
