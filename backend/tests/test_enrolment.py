"""Tests for enrolment endpoints (mocking DeepFace)."""
from __future__ import annotations
import io
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


def _create_test_user(client, auth_headers, email="enrolltest@example.com"):
    resp = client.post("/api/v1/users", json={
        "email": email,
        "full_name": "Enrol Test",
        "password": "TestPass123!",
    }, headers=auth_headers)
    return resp.json()["id"]


def test_consent_flow(client, auth_headers):
    user_id = _create_test_user(client, auth_headers, "consent@example.com")
    resp = client.post(f"/api/v1/enrolments/{user_id}/consent", json={
        "consent_given": True,
        "consent_text": "I agree to biometric enrolment",
    }, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["consent_given"] == True


def test_enrolment_status_not_enrolled(client, auth_headers):
    user_id = _create_test_user(client, auth_headers, "notenrolled@example.com")
    resp = client.get(f"/api/v1/enrolments/{user_id}", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_enrolled"] == False


def test_enrol_without_consent(client, auth_headers):
    user_id = _create_test_user(client, auth_headers, "noconsent@example.com")
    fake_image = io.BytesIO(b"fake-image-data")
    resp = client.post(
        f"/api/v1/enrolments/{user_id}/enrol",
        files={"file": ("face.jpg", fake_image, "image/jpeg")},
        headers=auth_headers,
    )
    assert resp.status_code == 400
    assert "Consent" in resp.json()["detail"]


@patch("backend.routers.enrolments.detect_face")
@patch("backend.routers.enrolments.assess_quality")
@patch("backend.routers.enrolments.generate_embedding")
@patch("backend.routers.enrolments.encrypt_embedding")
def test_enrol_success(mock_encrypt, mock_embed, mock_quality, mock_detect, client, auth_headers):
    from backend.schemas.enrolment import FaceDetectionResult, QualityResult
    mock_detect.return_value = FaceDetectionResult(face_detected=True, face_count=1, confidence=0.99, message="OK")
    mock_quality.return_value = QualityResult(quality_score=0.85, sharpness=0.8, brightness=0.5, face_size_ratio=0.2, is_acceptable=True, message="OK")
    mock_embed.return_value = np.zeros(512)
    mock_encrypt.return_value = b"encrypted_embedding_data"

    user_id = _create_test_user(client, auth_headers, "enrollsuccess@example.com")
    client.post(f"/api/v1/enrolments/{user_id}/consent", json={"consent_given": True}, headers=auth_headers)

    fake_image = io.BytesIO(b"\xff\xd8\xff" + b"\x00" * 100)
    resp = client.post(
        f"/api/v1/enrolments/{user_id}/enrol",
        files={"file": ("face.jpg", fake_image, "image/jpeg")},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert "quality_score" in resp.json()
