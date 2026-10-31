"""Coded API errors: rendered as `{"detail": message, "code": code}` by the application handler."""
from __future__ import annotations

from typing import Any, Optional


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: Optional[dict[str, str]] = None,
                 extra: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers, self.extra = status, code, message, headers, extra


class StepUpRequired(ApiError):
    def __init__(self, message: str = "Confirm your identity to continue", level: str = "standard"):
        super().__init__(403, "STEP_UP_REQUIRED", message, extra={"step_up_level": level})
