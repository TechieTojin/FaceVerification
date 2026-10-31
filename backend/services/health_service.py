"""Health and readiness — computed from real checks, never hardcoded."""
from __future__ import annotations

import platform
import sys
from datetime import datetime
from typing import Any, Optional

from backend.config import settings
from backend.core.events import bus
from backend.database import engine, ping_database
from backend.services.inference import runtime
from backend.services import maintenance_service


def migration_status() -> dict[str, Any]:
    try:
        from alembic.runtime.migration import MigrationContext
        from backend.migrations_util import head_revision
        with engine.connect() as conn:
            current = MigrationContext.configure(conn).get_current_revision()
        head = head_revision()
        return {"current": current, "head": head, "up_to_date": current == head}
    except Exception as exc:  # noqa: BLE001
        return {"current": None, "head": None, "up_to_date": False, "error": type(exc).__name__}


def readiness() -> dict[str, Any]:
    db_ok, db_ms, db_err = ping_database()
    mig = migration_status()
    rt = runtime.snapshot()
    checks = {
        "database": {"ok": db_ok, "latency_ms": round(db_ms, 1), "error": db_err or None},
        "migrations": {"ok": mig.get("up_to_date", False), **mig},
        "models": {"ok": rt["ready"], "state": rt["warmup_state"]},
    }
    return {"ready": all(c["ok"] for c in checks.values()), "checks": checks,
            "checked_at": datetime.utcnow().isoformat() + "Z"}


def system_health() -> dict[str, Any]:
    ready = readiness()
    rt = runtime.snapshot()
    m = maintenance_service.last_run
    uptime = (datetime.utcnow() - runtime.started_at).total_seconds()
    return {
        **ready,
        "version": settings.app_version,
        "environment": settings.app_env,
        "uptime_seconds": int(uptime),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "database_backend": "sqlite" if settings.is_sqlite else engine.dialect.name,
        "inference": rt,
        "realtime": {"sse_subscribers": bus.subscriber_count},
        "maintenance": {"last_run_at": m["at"].isoformat() + "Z" if m["at"] else None, "ok": m["ok"],
                        "result": m["result"], "error": m["error"]},
    }


def models_ready() -> Optional[bool]:
    return runtime.ready
