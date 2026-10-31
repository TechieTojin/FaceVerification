"""Test configuration — isolated environment.

Environment variables are set *before* any backend import because Settings()
is evaluated at import time. Tests never read the developer's .env secrets:
every value that matters is overridden here (env vars take precedence over
.env in pydantic-settings).

Inference uses the test-only FakeProvider (APP_ENV=test), which decodes
synthetic marker frames. Real-model tests live in test_realmodel.py.
"""
from __future__ import annotations

import os
import tempfile
import time

from cryptography.fernet import Fernet

_TMP = tempfile.mkdtemp(prefix="fv-test-")
os.environ.update({
    "APP_ENV": "test",
    "INFERENCE_PROVIDER": "fake",
    "SECRET_KEY": "test-only-secret-key-not-for-production-32ch!!",
    "DATABASE_URL": "sqlite:///" + os.path.join(_TMP, "test.db").replace("\\", "/"),
    "EMBEDDING_ENCRYPTION_KEY": Fernet.generate_key().decode(),
    "INITIAL_ADMIN_EMAIL": "admin@test.local",
    "INITIAL_ADMIN_PASSWORD": "Admin-Test-Pass-1",
    "INITIAL_ADMIN_NAME": "Test Admin",
    "BCRYPT_ROUNDS": "4",
    "BACKUP_DIR": os.path.join(_TMP, "backups"),
    "MAINTENANCE_INTERVAL_SECONDS": "3600",
    "WARMUP_MODELS": "true",
    "LOG_LEVEL": "WARNING",
    "INFERENCE_WORKERS": "2",
    "INFERENCE_QUEUE_LIMIT": "4",
})

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.config import settings  # noqa: E402
from backend.database import Base, SessionLocal  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models.organization import Membership, Organization  # noqa: E402
from backend.models.settings import PolicyVersion  # noqa: E402
from backend.models.user import User  # noqa: E402
from backend.services.inference import runtime  # noqa: E402

ADMIN_EMAIL = settings.initial_admin_email
ADMIN_PASSWORD = settings.initial_admin_password


@pytest.fixture(scope="session")
def app_client():
    with TestClient(app, raise_server_exceptions=False) as c:
        deadline = time.time() + 10
        while not runtime.ready and time.time() < deadline:
            time.sleep(0.05)
        assert runtime.ready, "fake provider warm-up did not complete"
        db = SessionLocal()
        admin = db.query(User).filter(User.email == ADMIN_EMAIL).one()
        org = db.query(Organization).filter(Organization.slug == "default").one()
        c.admin_snapshot = (admin.id, admin.hashed_password)
        c.default_org_id = org.id
        db.close()
        yield c


@pytest.fixture(autouse=True)
def clean_db(app_client):
    """Reset all data to: seeded platform admin + default organisation + its policy version 1."""
    from backend.services.biometric.templates import reset_crypto_cache
    keep = ("users", "policy_versions", "organizations", "memberships")
    db = SessionLocal()
    try:
        admin_id, admin_hash = app_client.admin_snapshot
        org_id = app_client.default_org_id
        for table in reversed(Base.metadata.sorted_tables):
            if table.name in keep:
                continue
            db.execute(table.delete())
        db.query(PolicyVersion).filter((PolicyVersion.version > 1) | (PolicyVersion.organization_id != org_id)).delete()
        db.query(Membership).filter((Membership.user_id != admin_id) | (Membership.organization_id != org_id)).delete()
        db.query(Membership).filter(Membership.user_id == admin_id).update({"role": "ADMIN", "status": "ACTIVE"})
        db.query(Organization).filter(Organization.id != org_id).delete()
        db.query(Organization).filter(Organization.id == org_id).update({"status": "ACTIVE",
                                                                         "name": "Default organization"})
        db.query(User).filter(User.id != admin_id).delete()
        db.query(User).filter(User.id == admin_id).update(
            {"hashed_password": admin_hash, "is_active": True, "password_changed_at": None,
             "is_platform_admin": True, "last_org_id": org_id})
        db.commit()
    finally:
        db.close()
    reset_crypto_cache()
    app_client.cookies.clear()
    yield


@pytest.fixture
def client(app_client):
    return app_client


@pytest.fixture
def db():
    s = SessionLocal()
    yield s
    s.close()
