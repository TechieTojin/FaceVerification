"""FaceVerification AI 2.1 — FastAPI backend entry point."""
from __future__ import annotations
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager

from backend.config import settings
from backend.database import create_tables, SessionLocal
from backend.middleware.logging import RequestLoggingMiddleware
from backend.routers import (
    auth, users, enrolments, verification,
    dashboard, analytics, security,
    settings as settings_router,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ── CSRF middleware ──────────────────────────────────────────────────────────

class CSRFMiddleware(BaseHTTPMiddleware):
    """Double-submit cookie CSRF protection for cookie-authenticated requests.

    Bearer-token requests are exempt (they are not browser-initiated and
    cannot be the target of CSRF attacks in the traditional sense).
    """
    _SAFE = frozenset({"GET", "HEAD", "OPTIONS"})
    _EXEMPT = frozenset({
        "/api/v1/auth/login",
        "/api/v1/auth/refresh",   # authenticated by HttpOnly refresh cookie
        "/api/v1/health",
    })

    async def dispatch(self, request: Request, call_next):
        if request.method in self._SAFE:
            return await call_next(request)
        if request.url.path in self._EXEMPT:
            return await call_next(request)

        # Only enforce CSRF for requests that carry the access cookie.
        # API clients using Bearer tokens are not browser-based and don't need CSRF.
        if not request.cookies.get("fv_access"):
            return await call_next(request)

        csrf_cookie = request.cookies.get(settings.csrf_cookie_name, "")
        csrf_header = request.headers.get("X-CSRF-Token", "")

        if not csrf_cookie or not csrf_header or csrf_cookie != csrf_header:
            return JSONResponse(
                status_code=403,
                content={"detail": "CSRF validation failed"},
            )

        return await call_next(request)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting FaceVerification AI 2.1 backend...")
    create_tables()

    db = SessionLocal()
    try:
        from backend.services.auth_service import seed_admin
        seed_admin(db)
    finally:
        db.close()

    log.info("Backend ready. Docs at http://localhost:8000/docs")
    yield
    log.info("Backend shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FaceVerification AI 2.1",
    description="Biometric identity verification platform",
    version="2.1.0",
    lifespan=lifespan,
)

# CORS — explicit origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-CSRF-Token"],
)

# CSRF protection for cookie-authenticated state-changing requests
app.add_middleware(CSRFMiddleware)

# Request logging
app.add_middleware(RequestLoggingMiddleware)

# Routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(enrolments.router)
app.include_router(verification.router)
app.include_router(dashboard.router)
app.include_router(analytics.router)
app.include_router(security.router)
app.include_router(settings_router.router)


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "FaceVerification AI 2.1",
        "version": "2.1.0",
    }
