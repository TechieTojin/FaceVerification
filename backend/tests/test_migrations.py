"""Migrations: fresh schema matches models; a populated v2.1 database upgrades safely."""
from __future__ import annotations

import os
import sqlite3
import tempfile
import uuid
from datetime import datetime

import numpy as np
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine

import backend.models  # noqa: F401
from backend.database import Base
from backend.migrations_util import alembic_config, upgrade_to_head
from backend.services.biometric.templates import LegacyTemplateError, _get_fernet, decrypt_vectors


def _url(path: str) -> str:
    return "sqlite:///" + path.replace("\\", "/")


def test_fresh_upgrade_matches_models():
    path = os.path.join(tempfile.mkdtemp(), "fresh.db")
    upgrade_to_head(_url(path), backup=False)
    eng = create_engine(_url(path))
    with eng.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata)
    eng.dispose()
    assert diff == []


def test_v21_database_upgrades_with_backup_and_data_mapping(tmp_path, monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "backup_dir", str(tmp_path / "bk"))
    path = str(tmp_path / "legacy.db")
    cfg = alembic_config(_url(path))
    command.upgrade(cfg, "0001_v21_baseline")
    con = sqlite3.connect(path)
    con.execute("DROP TABLE alembic_version")   # emulate a create_all() v2.1 database (no Alembic history)
    now = datetime.utcnow().isoformat(sep=" ")
    uid, sid = str(uuid.uuid4()), str(uuid.uuid4())
    legacy_blob = _get_fernet().encrypt(np.ones(512).tobytes())
    con.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?)",
                (uid, "Mixed@Case.Local", "Legacy", "x", "OPERATOR", 1, 0, now, now, None))
    con.execute("INSERT INTO biometric_enrolments VALUES (?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), uid, legacy_blob, 1, now, 0.8, now, 1, "ArcFace"))
    con.execute("INSERT INTO verification_sessions VALUES (?,?,?,?,?,?,?,?,?)",
                (sid, uid, None, now, now, 1, None, None, 1))
    con.execute("INSERT INTO verification_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), sid, uid, now, "SPOOF_DETECTED", None, None, 0.91, 1, 120.0, "x", "v1.0", 0))
    con.execute("INSERT INTO system_settings VALUES (?,?,?,?,?,?)",
                (str(uuid.uuid4()), "enable_liveness", "false", "", now, None))
    con.commit()
    con.close()

    info = upgrade_to_head(_url(path))
    assert info["migrated"] and info["from"] == "v2.1" and info["backup"] and os.path.exists(info["backup"])

    con = sqlite3.connect(path)
    assert con.execute("SELECT email FROM users").fetchone()[0] == "mixed@case.local"
    status, version, blob = con.execute(
        "SELECT status, template_version, embedding_encrypted FROM biometric_enrolments").fetchone()
    assert (status, version) == ("REENROLMENT_REQUIRED", 1)
    try:
        decrypt_vectors(blob)
        raise AssertionError("legacy template must not decode as v3")
    except LegacyTemplateError:
        pass
    assert con.execute("SELECT is_active, final_outcome FROM verification_sessions").fetchone() == (0, "SESSION_EXPIRED")
    assert con.execute("SELECT outcome, passive_label, passive_confidence FROM verification_events").fetchone() == (
        "LIVENESS_FAILED", "SPOOF", 0.91)
    pv = con.execute("SELECT values_json, change_summary FROM policy_versions").fetchone()
    assert '"passive_liveness_required": true' in pv[0]          # legacy "false" NOT silently applied
    assert "enable_liveness" in pv[1]
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "system_settings" not in tables and "user_roles" not in tables
    con.close()

    # v4: every row lands in the default organisation; roles move to memberships.
    con = sqlite3.connect(path)
    (org_id, slug), = con.execute("SELECT id, slug FROM organizations").fetchall()
    assert slug == "default"
    assert con.execute("SELECT organization_id, role, status FROM memberships WHERE user_id = ?", (uid,)).fetchone() == (
        org_id, "OPERATOR", "ACTIVE")
    cols = {r[1] for r in con.execute("PRAGMA table_info(users)")}
    assert "role" not in cols and "is_platform_admin" in cols
    for table in ("biometric_enrolments", "verification_sessions", "verification_events", "policy_versions"):
        assert con.execute(f"SELECT COUNT(*) FROM {table} WHERE organization_id IS NULL OR organization_id != ?",
                           (org_id,)).fetchone()[0] == 0, table
    # v2.1 sessions recorded no operator, so the owner reference is empty (they are closed anyway).
    assert con.execute("SELECT owner_kind, owner_ref, channel FROM verification_sessions").fetchone() == (
        "USER", "", "DESKTOP")
    con.close()


def _populated_v3(tmp_path) -> str:
    path = str(tmp_path / "v3.db")
    command.upgrade(alembic_config(_url(path)), "0002_v3_platform")
    con = sqlite3.connect(path)
    now = datetime.utcnow().isoformat(sep=" ")
    for i, role in enumerate(("ADMIN", "OPERATOR", "VIEWER")):
        con.execute("INSERT INTO users (id, email, full_name, hashed_password, role, is_active, is_verified, "
                    "created_at, updated_at) VALUES (?,?,?,?,?,1,0,?,?)",
                    (f"u{i}", f"u{i}@t.local", f"U{i}", "x", role, now, now))
    con.commit()
    con.close()
    return path


def test_v3_to_v4_upgrade_downgrade_round_trip(tmp_path, monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "backup_dir", str(tmp_path / "bk"))
    path = _populated_v3(tmp_path)
    info = upgrade_to_head(_url(path))
    assert info["from"] == "0002_v3_platform" and info["backup"] and os.path.exists(info["backup"])
    con = sqlite3.connect(path)
    assert con.execute("SELECT COUNT(*) FROM users WHERE is_platform_admin = 1").fetchone()[0] == 1   # v3 admin
    assert sorted(r[0] for r in con.execute("SELECT role FROM memberships")) == ["ADMIN", "OPERATOR", "VIEWER"]
    con.close()

    cfg = alembic_config(_url(path))
    command.downgrade(cfg, "0002_v3_platform")
    con = sqlite3.connect(path)
    assert sorted(r[0] for r in con.execute("SELECT role FROM users")) == ["ADMIN", "OPERATOR", "VIEWER"]
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "organizations" not in tables and "memberships" not in tables
    con.close()

    command.upgrade(cfg, "head")       # and forward again
    con = sqlite3.connect(path)
    assert con.execute("SELECT COUNT(*) FROM memberships").fetchone()[0] == 3
    con.close()


def test_downgrade_refuses_when_several_organisations_exist(tmp_path, monkeypatch):
    import pytest
    from backend.config import settings
    monkeypatch.setattr(settings, "backup_dir", str(tmp_path / "bk"))
    path = _populated_v3(tmp_path)
    upgrade_to_head(_url(path))
    con = sqlite3.connect(path)
    con.execute("INSERT INTO organizations (id, slug, name, status, created_at) VALUES ('o2', 'second', 'Second', "
                "'ACTIVE', ?)", (datetime.utcnow().isoformat(sep=" "),))
    con.commit()
    con.close()
    with pytest.raises(RuntimeError, match="more than one organisation"):
        command.downgrade(alembic_config(_url(path)), "0002_v3_platform")
    con = sqlite3.connect(path)   # nothing was changed
    assert con.execute("SELECT COUNT(*) FROM organizations").fetchone()[0] == 2
    con.close()
