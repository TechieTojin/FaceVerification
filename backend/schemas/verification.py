"""Verification request/response schemas."""
from __future__ import annotations
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from backend.models.verification import OutcomeEnum


class SessionCreateRequest(BaseModel):
    target_user_id: str


class SessionResponse(BaseModel):
    session_id: str
    target_user_id: str
    expires_at: datetime
    is_active: bool
    frame_count: int


class FrameResult(BaseModel):
    session_id: str
    outcome: OutcomeEnum
    action: str  # "retry", "reject", "verified", "lockout"
    explanation: str
    distance: Optional[float] = None
    threshold: Optional[float] = None
    liveness_score: Optional[float] = None
    face_count: int = 0
    inference_ms: float = 0.0
    attempts_remaining: int
    is_liveness_real: Optional[bool] = None


class VerificationEventResponse(BaseModel):
    id: str
    session_id: str
    user_id: Optional[str]
    timestamp: datetime
    outcome: OutcomeEnum
    distance: Optional[float]
    threshold: Optional[float]
    liveness_score: Optional[float]
    face_count: int
    inference_ms: Optional[float]
    explanation: Optional[str]

    class Config:
        from_attributes = True


class PaginatedEvents(BaseModel):
    total: int
    page: int
    page_size: int
    events: list[VerificationEventResponse]
