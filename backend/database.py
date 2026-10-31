"""SQLAlchemy engine and session management (SQLite or PostgreSQL)."""
from __future__ import annotations

import logging
import time

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import settings

log = logging.getLogger(__name__)


def _make_engine(url: str):
    kwargs: dict = {"echo": settings.debug, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 15}
    else:
        kwargs.update(pool_size=10, max_overflow=20, pool_recycle=1800)
    eng = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        # pysqlite's default transaction handling is kept on purpose: it opens the transaction at the
        # first write, so a read-then-write request waits on busy_timeout instead of failing with a
        # WAL snapshot conflict. SAVEPOINTs (capture terminal hooks) are only taken after the session
        # has written, i.e. inside that transaction.
        @event.listens_for(eng, "connect")
        def _sqlite_pragma(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=15000")
            cur.close()
    return eng


engine = _make_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)
Base = declarative_base()


def get_db():
    """FastAPI dependency yielding a session that is always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ping_database() -> tuple[bool, float, str]:
    """Return (ok, latency_ms, error) for readiness checks."""
    t0 = time.perf_counter()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, (time.perf_counter() - t0) * 1000, ""
    except Exception as exc:  # noqa: BLE001
        return False, (time.perf_counter() - t0) * 1000, type(exc).__name__
