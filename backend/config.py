"""Application configuration using pydantic-settings."""
from __future__ import annotations
import os
import secrets
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # App
    app_name: str = "FaceVerification AI 2.0"
    debug: bool = False

    # Database
    database_url: str = "sqlite:///./faceverification.db"

    # JWT
    secret_key: str = Field(default_factory=lambda: secrets.token_hex(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Biometric
    embedding_encryption_key: str = Field(default_factory=lambda: "")
    enable_liveness: bool = True
    arcface_model: str = "ArcFace"
    detector_backend: str = "opencv"
    max_face_distance: float = 0.68
    min_face_quality: float = 0.6

    # Session
    session_timeout_seconds: int = 120
    max_attempts_per_session: int = 5

    # Initial admin seed (used only when no users exist in the DB)
    initial_admin_email: str = "admin@faceverification.local"
    initial_admin_password: str = "Admin1234!"
    initial_admin_name: str = "System Administrator"

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def get_fernet_key(self) -> bytes:
        """Get or generate Fernet key for embedding encryption."""
        from cryptography.fernet import Fernet
        if self.embedding_encryption_key:
            key = self.embedding_encryption_key.encode()
            if len(key) == 44:
                return key
        # Generate and store
        return Fernet.generate_key()


settings = Settings()
