"""Application configuration — FaceVerification AI 2.1."""
from __future__ import annotations
import logging
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from pydantic import ConfigDict

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "FaceVerification AI 2.1"
    debug: bool = False

    # Database
    database_url: str = "sqlite:///./faceverification.db"

    # JWT / Access tokens  (short-lived — 15 min)
    secret_key: str = ""  # REQUIRED — set SECRET_KEY in .env
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15

    # Refresh tokens
    refresh_token_expire_days: int = 7
    refresh_token_cookie_name: str = "fv_refresh"

    # Cookies
    # Set cookie_secure=true in production (requires HTTPS).
    # Keep false on localhost (HTTP-only dev).
    cookie_secure: bool = False
    cookie_samesite: str = "strict"

    # CSRF double-submit cookie
    csrf_cookie_name: str = "fv_csrf"

    # Rate limiting (login endpoint)
    login_rate_limit_per_ip: int = 20        # per window per IP
    login_rate_limit_per_account: int = 10   # per window per account
    login_rate_limit_window_seconds: int = 600  # 10-minute window

    # Account lockout
    login_lockout_threshold: int = 5              # consecutive failures
    login_lockout_duration_seconds: int = 300     # 5 minutes

    # Biometric
    embedding_encryption_key: str = Field(default_factory=lambda: "")
    enable_liveness: bool = True
    arcface_model: str = "ArcFace"
    detector_backend: str = "opencv"
    max_face_distance: float = 0.68
    min_face_quality: float = 0.6

    # Biometric session
    session_timeout_seconds: int = 120
    max_attempts_per_session: int = 5

    # Initial admin seed (used only when DB has no users)
    initial_admin_email: str = "admin@faceverification.local"
    initial_admin_password: str = ""  # Set INITIAL_ADMIN_PASSWORD in .env
    initial_admin_name: str = "System Administrator"

    # CORS — restrict to configured trusted origins
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    @model_validator(mode="after")
    def require_secret_key(self) -> "Settings":
        if not self.secret_key:
            raise ValueError(
                "SECRET_KEY is not set.\n"
                "Generate one: python -c \"import secrets; print(secrets.token_hex(32))\"\n"
                "Then add SECRET_KEY=<value> to your .env file."
            )
        if len(self.secret_key) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters.")
        return self

    def get_fernet_key(self) -> bytes:
        from cryptography.fernet import Fernet
        if self.embedding_encryption_key:
            key = self.embedding_encryption_key.encode()
            if len(key) == 44:
                return key
        return Fernet.generate_key()


settings = Settings()
