"""Alembic helpers: programmatic config, safe upgrade with SQLite backup."""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

from backend.config import settings

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
BASELINE_REVISION = "0001_v21_baseline"


def alembic_config(url: Optional[str] = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", (url or settings.database_url).replace("%", "%%"))
    return cfg


def head_revision() -> str:
    return ScriptDirectory.from_config(alembic_config()).get_current_head()


def _sqlite_path(url: str) -> Optional[Path]:
    if not url.startswith("sqlite:///") or url.endswith(":memory:"):
        return None
    return Path(url.replace("sqlite:///", "", 1))


def backup_sqlite(url: str, label: str) -> Optional[Path]:
    """Consistent online copy (includes WAL contents) via the sqlite3 backup API."""
    path = _sqlite_path(url)
    if path is None or not path.exists():
        return None
    out_dir = Path(settings.backup_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{path.stem}-{datetime.utcnow():%Y%m%dT%H%M%SZ}-{label}.db"
    src = sqlite3.connect(str(path))
    try:
        dst = sqlite3.connect(str(dest))
        with dst:
            src.backup(dst)
        dst.close()
    finally:
        src.close()
    log.warning("Database backed up to %s before migration", dest)
    return dest


def current_revision(url: Optional[str] = None) -> tuple[Optional[str], bool]:
    """Returns (revision, has_legacy_tables)."""
    eng = create_engine(url or settings.database_url)
    try:
        with eng.connect() as conn:
            rev = MigrationContext.configure(conn).get_current_revision()
            tables = set(inspect(conn).get_table_names())
        return rev, ("users" in tables and rev is None)
    finally:
        eng.dispose()


def upgrade_to_head(url: Optional[str] = None, backup: bool = True) -> dict:
    url = url or settings.database_url
    cfg = alembic_config(url)
    head = ScriptDirectory.from_config(cfg).get_current_head()
    rev, legacy = current_revision(url)
    if rev == head:
        return {"migrated": False, "revision": rev}
    backup_path = backup_sqlite(url, f"pre-{head}") if backup else None
    if legacy:
        log.warning("Existing pre-Alembic (v2.1) schema detected; stamping %s", BASELINE_REVISION)
        command.stamp(cfg, BASELINE_REVISION)
    command.upgrade(cfg, "head")
    log.info("Database migrated %s → %s", rev or ("v2.1" if legacy else "empty"), head)
    return {"migrated": True, "from": rev or ("v2.1" if legacy else None), "to": head,
            "backup": str(backup_path) if backup_path else None}
