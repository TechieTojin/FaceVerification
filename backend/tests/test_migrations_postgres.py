"""Opt-in: run the Alembic migrations against a real PostgreSQL server.

Point FV_TEST_POSTGRES_URL at an EMPTY, throwaway database (everything in it is dropped):

    $env:FV_TEST_POSTGRES_URL = "postgresql+psycopg://user:password@localhost:5432/fv_migration_test"
    .venv\\Scripts\\python -m pytest backend\\tests\\test_migrations_postgres.py -v
"""
from __future__ import annotations

import os

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text

import backend.models  # noqa: F401
from backend.database import Base
from backend.migrations_util import alembic_config

PG_URL = os.environ.get("FV_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="set FV_TEST_POSTGRES_URL to an empty throwaway PostgreSQL database")


def _reset(url: str) -> None:
    eng = create_engine(url)
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    eng.dispose()


def test_postgres_upgrade_matches_models_and_round_trips():
    _reset(PG_URL)
    cfg = alembic_config(PG_URL)
    command.upgrade(cfg, "0002_v3_platform")
    eng = create_engine(PG_URL)
    with eng.begin() as conn:
        conn.execute(text("INSERT INTO users (id, email, full_name, hashed_password, role, is_active, is_verified, "
                          "created_at, updated_at) VALUES ('u1', 'a@t.local', 'A', 'x', 'ADMIN', true, false, now(), now())"))
    command.upgrade(cfg, "head")
    with eng.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata)
        assert diff == []
        assert conn.execute(text("SELECT role FROM memberships WHERE user_id = 'u1'")).scalar() == "ADMIN"
        assert conn.execute(text("SELECT is_platform_admin FROM users WHERE id = 'u1'")).scalar() is True
    command.downgrade(cfg, "0002_v3_platform")
    with eng.connect() as conn:
        assert conn.execute(text("SELECT role FROM users WHERE id = 'u1'")).scalar() == "ADMIN"
    command.upgrade(cfg, "head")
    eng.dispose()
    _reset(PG_URL)
