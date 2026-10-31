"""Shared test helpers (deterministic, no real biometric data)."""
from __future__ import annotations

import itertools
from datetime import datetime
from typing import Any, Callable, Optional

from fastapi.testclient import TestClient

from backend.database import SessionLocal
from backend.models.biometric import BiometricEnrolment, EnrolmentStatus
from backend.models.organization import Organization
from backend.models.user import RoleEnum, User
from backend.services.auth_service import create_user
from backend.services.biometric.providers import fake_identity_vector, make_fake_frame
from backend.services.biometric.templates import encrypt_vectors

PASSWORD = "Str0ng-Test-Pass!"
_nonce = itertools.count(1)


def default_org_id() -> str:
    db = SessionLocal()
    try:
        return db.query(Organization).filter(Organization.slug == "default").one().id
    finally:
        db.close()


def make_org(name: str = "Second Org", slug: Optional[str] = None) -> str:
    from backend.services.policy_service import ensure_policy_seeded
    db = SessionLocal()
    try:
        org = Organization(name=name, slug=slug or f"org-{next(_nonce)}", status="ACTIVE")
        db.add(org)
        db.commit()
        ensure_policy_seeded(db, org.id)
        return org.id
    finally:
        db.close()


def add_member(user_id: str, org_id: str, role: RoleEnum = RoleEnum.OPERATOR) -> None:
    from backend.models.organization import Membership
    db = SessionLocal()
    try:
        db.add(Membership(organization_id=org_id, user_id=user_id, role=role, status="ACTIVE"))
        db.commit()
    finally:
        db.close()


def make_user(role: RoleEnum = RoleEnum.OPERATOR, email: Optional[str] = None, name: str = "Test User",
              password: str = PASSWORD, active: bool = True, org_id: Optional[str] = None) -> User:
    db = SessionLocal()
    try:
        u = create_user(db, email or f"{role.value.lower()}-{next(_nonce)}@test.local", name, password, role,
                        org_id=org_id or default_org_id())
        if not active:
            u.is_active = False
            db.commit()
        db.refresh(u)
        db.expunge(u)
        return u
    finally:
        db.close()


def login(client: TestClient, email: str, password: str = PASSWORD) -> dict[str, str]:
    """Bearer-token login (cookies cleared so CSRF middleware does not apply)."""
    r = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    client.cookies.clear()
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def admin_headers(client: TestClient) -> dict[str, str]:
    from backend.config import settings
    return login(client, settings.initial_admin_email, settings.initial_admin_password)


def admin_id() -> str:
    from backend.config import settings
    db = SessionLocal()
    try:
        return db.query(User).filter(User.email == settings.initial_admin_email).one().id
    finally:
        db.close()


def enrol_direct(user_id: str, identity: int = 1, consent: bool = True, version: int = 2,
                 org_id: Optional[str] = None) -> None:
    """Install an active template for `user_id` (in `org_id`, default organisation) without the capture flow."""
    org_id = org_id or default_org_id()
    db = SessionLocal()
    try:
        enr = (db.query(BiometricEnrolment)
               .filter(BiometricEnrolment.user_id == user_id, BiometricEnrolment.organization_id == org_id).first())
        if enr is None:
            enr = BiometricEnrolment(user_id=user_id, organization_id=org_id)
            db.add(enr)
        enr.embedding_encrypted = encrypt_vectors([fake_identity_vector(identity)], "ArcFace")
        enr.status = EnrolmentStatus.ACTIVE
        enr.template_version = version
        enr.embedding_dim = 512
        enr.sample_count = 3
        enr.quality_score = 0.9
        enr.consent_given = consent
        enr.consent_timestamp = datetime.utcnow()
        enr.enrolled_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()


def frame(**kw: Any) -> bytes:
    kw.setdefault("nonce", next(_nonce))
    return make_fake_frame(**kw)


def frame_for_step(step: str, identity: int, phase: int = 0, **overrides: Any) -> bytes:
    """A frame that satisfies `step` (BLINK alternates open → closed by phase)."""
    params: dict[str, Any] = {"identity": identity, "yaw_deg": 0.0, "blink": 0.0}
    if step == "TURN_LEFT":
        params["yaw_deg"] = 25.0
    elif step == "TURN_RIGHT":
        params["yaw_deg"] = -25.0
    elif step == "BLINK":
        params["blink"] = 0.0 if phase % 2 == 0 else 1.0
    params.update(overrides)
    return frame(**params)


def submit(client: TestClient, headers: dict, session: dict, data: bytes):
    return client.post(f"/api/v1/sessions/{session['session_id']}/frames", headers=headers,
                       data={"seq": str(session["frame_seq"] + 1), "frame_token": session["frame_token"]},
                       files={"file": ("f.jpg", data, "image/jpeg")})


def drive(client: TestClient, headers: dict, session: dict, identity: int = 1,
          frame_fn: Optional[Callable[[dict, int], bytes]] = None, max_frames: int = 80) -> tuple[dict, list[dict]]:
    """Answer the server's challenge until the session reaches a terminal state."""
    history: list[dict] = []
    state = session
    phase = 0
    for _ in range(max_frames):
        ch = state.get("challenge")
        if state.get("status") != "ACTIVE" or ch is None:
            break
        data = frame_fn(state, phase) if frame_fn else frame_for_step(ch["type"], identity, phase)
        phase += 1
        r = submit(client, headers, state, data)
        assert r.status_code == 200, r.text
        state = r.json()
        history.append(state)
        if state["frame"] and state["frame"]["terminal"]:
            break
    return state, history


def start_verify(client: TestClient, headers: dict, target_id: str) -> dict:
    r = client.post("/api/v1/sessions", json={"target_user_id": target_id}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()
