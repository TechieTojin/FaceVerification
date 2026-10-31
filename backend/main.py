"""FaceVerification 4.0 — FastAPI application."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402

from backend.config import settings  # noqa: E402
from backend.core.logging import RequestContextMiddleware, setup_logging  # noqa: E402
from backend.database import SessionLocal  # noqa: E402
from backend.core.errors import ApiError  # noqa: E402
from backend.routers import (  # noqa: E402
    auth, auth_factors, dashboard, devices, documents, enrolments, evaluation, history, integrations, notifications,
    organizations, pairings, reviews, security, system, users, verification,
)
from backend.routers import settings as policy_router  # noqa: E402
from backend.services.capture_service import CaptureError  # noqa: E402
from backend.services.document_service import DocumentError  # noqa: E402
from backend.services.enrolment_service import EnrolmentDenied  # noqa: E402
from backend.services.inference import build_provider, runtime  # noqa: E402

setup_logging(settings.log_level, settings.log_format)

from backend.services.wiring import register_all  # noqa: E402

register_all()
log = logging.getLogger(__name__)


class CSRFMiddleware(BaseHTTPMiddleware):
    """Double-submit-cookie CSRF check for cookie-authenticated state changes.

    Bearer-token clients are exempt: browsers never attach Authorization headers
    cross-site, so they cannot be driven by CSRF.
    """

    _SAFE = frozenset({"GET", "HEAD", "OPTIONS"})
    # Pre-authentication endpoints: they never act on cookie credentials.
    _EXEMPT = frozenset({"/api/v1/auth/login", "/api/v1/auth/refresh", "/api/v1/auth/passkey/options",
                         "/api/v1/auth/passkey/verify", "/api/v1/auth/mfa/verify", "/api/v1/auth/mfa/webauthn/options",
                         "/api/v1/auth/mfa/email-otp", "/api/v1/auth/mfa/enrol/totp",
                         "/api/v1/auth/mfa/enrol/totp/confirm", "/api/v1/auth/login/face/session",
                         "/api/v1/auth/login/face/complete"})

    # Header-credential endpoints (paired phones): cookies are never consulted.
    # Integration clients authenticate with API keys, never cookies.
    # Integration clients (API keys) and devices (request signatures) never use cookies either.
    _EXEMPT_PREFIXES = ("/api/v1/remote/", "/api/v1/integrations/sessions", "/api/v1/integrations/assertions/",
                        "/api/v1/device-api/")

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if request.method in self._SAFE or path in self._EXEMPT or path.startswith(self._EXEMPT_PREFIXES):
            return await call_next(request)
        if not request.cookies.get("fv_access"):
            return await call_next(request)
        cookie = request.cookies.get(settings.csrf_cookie_name, "")
        header = request.headers.get("X-CSRF-Token", "")
        if not cookie or not header or cookie != header:
            return JSONResponse(status_code=403, content={"detail": "CSRF validation failed", "code": "CSRF_FAILED"})
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        if settings.cookie_secure:
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response


def _record_models(results: list[dict]) -> None:
    from backend.models.documents import ModelMetadata
    db = SessionLocal()
    try:
        for m in results:
            row = db.query(ModelMetadata).filter(ModelMetadata.name == m["name"]).first()
            if row is None:
                row = ModelMetadata(name=m["name"], task=m.get("task", "unknown"))
                db.add(row)
            row.status = m.get("status", "UNKNOWN")
            row.load_ms = m.get("load_ms")
            row.error = m.get("error")
            row.file_name = m.get("file_name", row.file_name)
            row.sha256 = m.get("sha256", row.sha256)
            row.embedding_dim = m.get("embedding_dim", row.embedding_dim)
            row.updated_at = datetime.utcnow()
        db.commit()
    except Exception:  # noqa: BLE001
        log.exception("Could not persist model metadata")
    finally:
        db.close()


async def _maintenance_loop() -> None:
    from backend.services.maintenance_service import run_maintenance
    while True:
        await asyncio.sleep(settings.maintenance_interval_seconds)
        await run_in_threadpool(run_maintenance)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s (%s)", settings.app_name, settings.app_env)
    from backend.migrations_util import head_revision, current_revision, upgrade_to_head
    if settings.auto_migrate:
        info = await run_in_threadpool(upgrade_to_head)
        if info.get("migrated"):
            log.warning("Schema migrated: %s", info)
    else:
        rev, _ = current_revision()
        if rev != head_revision():
            log.error("Database revision %s is not head %s; run `python -m backend.manage migrate`", rev, head_revision())

    db = SessionLocal()
    try:
        from backend.models.organization import Organization
        from backend.services.auth_service import seed_admin
        from backend.services.policy_service import ensure_policy_seeded
        seed_admin(db)
        for org in db.query(Organization).all():
            ensure_policy_seeded(db, org.id)
    finally:
        db.close()

    runtime.configure(build_provider())
    if settings.warmup_models:
        runtime.warmup_async(on_done=_record_models)
    task = asyncio.create_task(_maintenance_loop())
    log.info("API ready — docs at /docs")
    try:
        yield
    finally:
        task.cancel()
        runtime.shutdown()
        log.info("Shut down cleanly")


app = FastAPI(title=settings.app_name, version=settings.app_version,
              description="Biometric identity verification platform", lifespan=lifespan)


@app.exception_handler(CaptureError)
async def _capture_error(_: Request, exc: CaptureError):
    body = {"detail": exc.message, "code": exc.code}
    if exc.payload:
        body["session"] = exc.payload
    return JSONResponse(status_code=exc.status, content=_jsonable(body), headers=exc.headers)


@app.exception_handler(ApiError)
async def _api_error(_: Request, exc: ApiError):
    body = {"detail": exc.message, "code": exc.code, **(exc.extra or {})}
    return JSONResponse(status_code=exc.status, content=_jsonable(body), headers=exc.headers)


@app.exception_handler(EnrolmentDenied)
async def _enrol_error(_: Request, exc: EnrolmentDenied):
    return JSONResponse(status_code=exc.status, content={"detail": exc.message, "code": exc.code})


@app.exception_handler(DocumentError)
async def _doc_error(_: Request, exc: DocumentError):
    return JSONResponse(status_code=exc.status, content={"detail": exc.message, "code": exc.code})


def _jsonable(obj):
    from fastapi.encoders import jsonable_encoder
    return jsonable_encoder(obj)


# Middleware: last added runs first (outermost).
app.add_middleware(CSRFMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                   allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
                   allow_headers=["Content-Type", "Authorization", "X-CSRF-Token", "X-Request-ID", "X-Login-Token",
                                  "X-Pairing-Secret", "Idempotency-Key"],
                   expose_headers=["X-Request-ID", "Retry-After", "Content-Disposition"])
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestContextMiddleware)

for r in (system, auth, auth_factors, users, enrolments, verification, history, dashboard, evaluation, security,
          policy_router, documents, organizations, notifications, pairings, reviews, integrations, devices):
    app.include_router(r.router)
app.include_router(organizations.public)
app.include_router(pairings.remote)
app.include_router(devices.device_api)
