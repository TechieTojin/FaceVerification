"""SQLAlchemy database engine and session management."""
from __future__ import annotations
import logging
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.config import settings

log = logging.getLogger(__name__)

# Create engine
connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=settings.debug,
)

# Enable WAL mode for SQLite
if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency for database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables from models."""
    Base.metadata.create_all(bind=engine)
    log.info("Database tables created/verified")
