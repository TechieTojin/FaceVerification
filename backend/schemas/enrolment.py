"""Enrolment request/response schemas."""
from __future__ import annotations
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ConsentRequest(BaseModel):
    consent_given: bool
    consent_text: Optional[str] = None


class EnrolmentResponse(BaseModel):
    id: str
    user_id: str
    consent_given: bool
    consent_timestamp: Optional[datetime] = None
    quality_score: float
    enrolled_at: Optional[datetime] = None
    is_active: bool
    model_name: str

    class Config:
        from_attributes = True


class EnrolmentStatus(BaseModel):
    user_id: str
    is_enrolled: bool
    consent_given: bool
    quality_score: float
    enrolled_at: Optional[datetime] = None
    model_name: Optional[str] = None


class QualityResult(BaseModel):
    quality_score: float
    sharpness: float
    brightness: float
    face_size_ratio: float
    is_acceptable: bool
    message: str


class FaceDetectionResult(BaseModel):
    face_detected: bool
    face_count: int
    confidence: float
    message: str
