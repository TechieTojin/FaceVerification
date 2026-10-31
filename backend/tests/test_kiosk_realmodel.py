"""Opt-in: the offline kiosk making decisions with the REAL models (MediaPipe + MiniFASNet + ArcFace) locally.

    $env:RUN_REAL_MODEL_TESTS=1; .venv\\Scripts\\python -m pytest backend\\tests\\test_kiosk_realmodel.py -s

The kiosk is provisioned through the real server code path (signed requests, encrypted bundle); its central
connection is then cut and decisions are made by the kiosk process's own inference. Active liveness is disabled
for this test because still photographs cannot blink or turn — that check is covered by the fake-provider suite.
Two photos per person say nothing about accuracy.
"""
from __future__ import annotations

import cv2
import pytest

from backend.models.user import RoleEnum
from backend.services.biometric.templates import encrypt_vectors
from backend.tests.helpers import admin_headers, make_user
from backend.tests.test_devices_kiosk import Transport, _provision_kiosk
from backend.tests.test_realmodel import LANDMARKER, framed, pytestmark  # noqa: F401  (same opt-in gate)
from kiosk import client as kclient


@pytest.fixture(scope="module")
def real_provider():
    from backend.services.biometric.providers import DeepFaceProvider
    p = DeepFaceProvider(LANDMARKER)
    assert all(m["status"] == "READY" for m in p.warmup())
    return p


def _enrol_real(user_id: str, provider, image: str) -> None:
    from backend.database import SessionLocal
    from backend.models.biometric import BiometricEnrolment, EnrolmentStatus
    from backend.tests.helpers import default_org_id
    bgr = framed(image)
    face = provider.analyze(bgr).faces[0]
    vec = provider.embed(bgr, face)
    db = SessionLocal()
    org = default_org_id()
    enr = BiometricEnrolment(organization_id=org, user_id=user_id, status=EnrolmentStatus.ACTIVE,
                             embedding_encrypted=encrypt_vectors([vec], "ArcFace"), template_version=2,
                             embedding_dim=int(vec.shape[0]), sample_count=1, quality_score=0.9, consent_given=True)
    db.add(enr)
    db.commit()
    db.close()


def _drive_photo(agent, user_id: str, image: str) -> dict:
    ok, buf = cv2.imencode(".jpg", framed(image), [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    data = buf.tobytes()
    view = agent.start_session(user_id)
    for i in range(12):
        if view["status"] != "ACTIVE":
            break
        noisy = bytearray(data)
        noisy[-3] = (noisy[-3] + i + 1) % 256        # distinct bytes per frame (the kiosk rejects identical frames)
        view = agent.submit_frame(view["session_id"], view["frame_seq"] + 1, view["frame_token"], bytes(noisy))
    return view


def test_kiosk_decides_offline_with_real_models(client, monkeypatch, tmp_path, real_provider):
    transport = Transport(client)
    monkeypatch.setattr(kclient, "_request", transport)
    monkeypatch.setenv("KIOSK_PASSPHRASE", "test-only-kiosk-passphrase")
    assert client.put("/api/v1/policy", json={"changes": {"active_liveness_required": False}, "confirm": True,
                                              "reason": "still-photo kiosk test"},
                      headers=admin_headers(client)).status_code == 200
    bush = make_user(RoleEnum.VIEWER, name="George Bush")
    _enrol_real(bush.id, real_provider, "ref_bush_1.jpg")
    dev, store, agent = _provision_kiosk(client, transport, tmp_path, [bush])
    agent.provider = real_provider                              # the kiosk's own local inference (real models)

    transport.online = False
    assert agent.sync(False)["error"] == "SERVER_UNREACHABLE"         # the kiosk notices the outage
    genuine = _drive_photo(agent, bush.id, "bush_valid_1.jpg")
    impostor = _drive_photo(agent, bush.id, "powell_valid_0.jpg")
    print("offline genuine:", genuine["final_outcome"], "impostor:", impostor["final_outcome"])
    assert impostor["final_outcome"] in ("NOT_MATCHED", "LIVENESS_FAILED")       # never VERIFIED
    # Still photographs: the kiosk's local MiniFASNet may (correctly) reject them as presentation attacks.
    assert genuine["final_outcome"] in ("VERIFIED", "LIVENESS_FAILED", "NOT_MATCHED")
    events = store.recent_events()
    assert [e["payload"]["inference"] for e in events[:2]] == ["deepface", "deepface"]
    assert all(e["payload"]["offline"] for e in events[:2])

    transport.online = True
    assert agent.sync(False)["uploaded"] == 2
