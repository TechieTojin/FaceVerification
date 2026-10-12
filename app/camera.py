"""VideoCapture lifecycle manager."""
from __future__ import annotations

import logging
import sys

import cv2

log = logging.getLogger(__name__)


class CameraManager:
    """Open, configure, and release a cv2.VideoCapture."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> cv2.VideoCapture:
        """Open the camera; raises SystemExit on failure."""
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            log.error(
                "Cannot open camera (index %d). "
                "Check connection or try --camera 1.",
                self._index,
            )
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Camera %d opened at %dx%d", self._index, actual_w, actual_h)
        self._cap = cap
        return cap

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Camera %d released.", self._index)

    def __enter__(self) -> cv2.VideoCapture:
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.release()
