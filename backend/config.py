"""Application configuration — FaceVerification 4.0.

Process-level configuration comes from the environment / `.env`.
Runtime verification *policy* (thresholds, attempts, liveness …) lives in the
database and is managed by `backend.services.policy_service`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "FaceVerification 4.0"
    app_version: str = "4.0.0"
    app_env: Literal["development", "test", "production"] = "development"
    debug: bool = False
    log_format: Literal["text", "json"] = "text"
    log_level: str = "INFO"

    # Database
    database_url: str = "sqlite:///./faceverification.db"
    auto_migrate: bool = True          # apply Alembic migrations at startup (backs up SQLite first)
    backup_dir: str = "backups"

    # Password hashing cost (bcrypt log2 rounds). Tests may lower it; production floor is 10.
    bcrypt_rounds: int = 12

    # JWT / access tokens
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15

    # Refresh tokens
    refresh_token_expire_days: int = 7
    refresh_token_cookie_name: str = "fv_refresh"

    # Cookies
    cookie_secure: bool = False
    cookie_samesite: Literal["strict", "lax", "none"] = "strict"
    csrf_cookie_name: str = "fv_csrf"

    # Login rate limiting / lockout
    login_rate_limit_per_ip: int = 20
    login_rate_limit_per_account: int = 10
    login_rate_limit_window_seconds: int = 600
    login_lockout_threshold: int = 5
    login_lockout_duration_seconds: int = 300

    # Biometric crypto
    embedding_encryption_key: str = Field(default_factory=lambda: "")

    # Inference runtime
    inference_provider: Literal["deepface", "fake"] = "deepface"
    inference_workers: int = 2
    inference_queue_limit: int = 8
    models_dir: str = str(REPO_ROOT / "models")
    landmarker_model_file: str = "face_landmarker.task"
    warmup_models: bool = True

    # Background maintenance
    maintenance_interval_seconds: int = 60

    # Document authenticity provider ("none" = not available; never simulated)
    document_auth_provider: Literal["none"] = "none"

    # Initial admin seed (used only when DB has no users)
    initial_admin_email: str = "admin@faceverification.local"
    initial_admin_password: str = ""
    initial_admin_name: str = "System Administrator"

    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Public URL of the web UI, used in QR codes, hosted verification links and emails.
    public_base_url: str = "http://localhost:5173"

    # WebAuthn relying party. The RP ID must be the registrable domain of every allowed origin;
    # browsers do not allow IP addresses as RP IDs.
    webauthn_rp_id: str = "localhost"
    webauthn_rp_name: str = "FaceVerification"
    webauthn_origins: list[str] = ["http://localhost:5173"]

    # Outbound email (notifications, invitations, email one-time codes). Empty host = email unavailable.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = True
    smtp_from: str = "FaceVerification <no-reply@faceverification.local>"
    smtp_timeout_seconds: float = 10.0

    # Integration API
    api_rate_limit_per_minute: int = 60
    webhook_timeout_seconds: float = 5.0
    webhook_allow_private_targets: bool = False     # allow http/localhost/private-IP webhook URLs (dev/test)
    assertion_ttl_seconds: int = 300

    # Cross-device pairing
    pairing_ttl_seconds: int = 180

    # Offline kiosks: hard ceiling for any kiosk's offline validity period
    offline_max_validity_hours: int = 72

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if not self.secret_key:
            raise ValueError(
                "SECRET_KEY is not set.\n"
                "Generate one: python -c \"import secrets; print(secrets.token_hex(32))\"\n"
                "Then add SECRET_KEY=<value> to your .env file."
            )
        if len(self.secret_key) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters.")
        if self.inference_provider == "fake" and self.app_env != "test":
            raise ValueError(
                "INFERENCE_PROVIDER=fake is only permitted when APP_ENV=test. "
                "Production and development always use real model inference."
            )
        if self.bcrypt_rounds < 10 and self.app_env != "test":
            raise ValueError("BCRYPT_ROUNDS below 10 is only permitted when APP_ENV=test.")
        if self.offline_max_validity_hours < 1 or self.offline_max_validity_hours > 168:
            raise ValueError("OFFLINE_MAX_VALIDITY_HOURS must be between 1 and 168.")
        if self.app_env == "production" and not self.cookie_secure:
            log.warning("APP_ENV=production with COOKIE_SECURE=false — cookies will be sent over HTTP.")
        return self

    @property
    def email_available(self) -> bool:
        return bool(self.smtp_host)

    @property
    def landmarker_model_path(self) -> Path:
        return Path(self.models_dir) / self.landmarker_model_file

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def get_fernet_key(self) -> bytes:
        """Return the Fernet key bytes, or raise RuntimeError if not configured.

        Never falls back to a generated key: an ephemeral key would make stored
        templates permanently unreadable after a restart (silent data loss).
        """
        if not self.embedding_encryption_key:
            raise RuntimeError(
                "EMBEDDING_ENCRYPTION_KEY is not set. "
                "Generate one: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\" "
                "and add it to .env before enrolling users."
            )
        key = self.embedding_encryption_key.encode()
        if len(key) != 44:
            raise RuntimeError(
                f"EMBEDDING_ENCRYPTION_KEY has invalid length ({len(key)} chars); "
                "a Fernet key must be exactly 44 base64-url characters. "
                "Regenerate with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )
        return key


settings = Settings()
