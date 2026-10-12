"""Pytest fixtures — v2.1.

SECRET_KEY must be set before any backend import because Settings() is
evaluated at module load time.  This MUST remain the first executable line.
"""
from __future__ import annotations
import os

os.environ.setdefault("SECRET_KEY", "test-only-secret-key-not-for-production-32ch!!")

import pytest
from backend.config import settings
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base, get_db
from backend.main import app
from backend.services.auth_service import seed_admin

TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="session")
def session_factory(db_engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=db_engine)


@pytest.fixture(scope="session", autouse=True)
def seeded_db(session_factory):
    session = session_factory()
    seed_admin(session)
    session.close()


@pytest.fixture(autouse=True, scope="function")
def clean_transient_state(session_factory):
    """
    Clear rate-limiting and refresh-token data before every test.

    record_login_attempt() calls db.commit(), so rolled-back sessions do not
    undo those rows.  Clearing at the start of each test ensures a clean slate
    without requiring complex savepoint machinery.
    """
    from backend.models.auth import LoginAttempt, RefreshToken
    s = session_factory()
    s.query(LoginAttempt).delete(synchronize_session=False)
    s.query(RefreshToken).delete(synchronize_session=False)
    s.commit()
    s.close()
    yield


@pytest.fixture(scope="function")
def db_session(session_factory, clean_transient_state):
    session = session_factory()
    yield session
    session.rollback()
    session.close()


@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def admin_token(client):
    """Log in and return the Bearer access token (backwards-compat).

    Clears session cookies after login so that tests using only Bearer
    auth headers are not affected by CSRF middleware (which only activates
    when the fv_access cookie is present).
    """
    resp = client.post("/api/v1/auth/login", json={
        "email": settings.initial_admin_email,
        "password": settings.initial_admin_password,
    })
    assert resp.status_code == 200, f"Admin login failed ({resp.status_code}): {resp.text}"
    token = resp.json()["access_token"]
    # Remove cookies so CSRF middleware is not triggered on Bearer-only requests
    client.cookies.clear()
    return token


@pytest.fixture(scope="function")
def auth_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture(scope="function")
def logged_in_client(client):
    """A client that has completed the full cookie login flow."""
    resp = client.post("/api/v1/auth/login", json={
        "email": settings.initial_admin_email,
        "password": settings.initial_admin_password,
    })
    assert resp.status_code == 200, f"Cookie login failed ({resp.status_code}): {resp.text}"
    return client


@pytest.fixture(scope="function")
def csrf_headers(logged_in_client):
    csrf = logged_in_client.cookies.get("fv_csrf", "")
    return {"X-CSRF-Token": csrf}
