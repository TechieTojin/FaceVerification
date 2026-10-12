"""FaceVerification AI 2.0 — FastAPI Backend Entry Point."""
from __future__ import annotations
import logging
import sys
import os

# Add project root to path so backend package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.config import settings
from backend.database import create_tables, SessionLocal
from backend.middleware.logging import RequestLoggingMiddleware
from backend.routers import auth, users, enrolments, verification, dashboard, analytics, security, settings as settings_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    log.info("Starting FaceVerification AI 2.0 backend...")
    create_tables()

    # Seed admin user if needed
    db = SessionLocal()
    try:
        from backend.services.auth_service import seed_admin
        seed_admin(db)
    finally:
        db.close()

    log.info("Backend ready. Docs at http://localhost:8000/docs")
    yield
    log.info("Backend shutting down.")


app = FastAPI(
    title="FaceVerification AI 2.0",
    description="Biometric identity verification platform",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        "service": "FaceVerification AI 2.0",
        "version": "2.0.0",
    }
