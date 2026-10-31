"""Prepare an isolated environment before any `backend` module is imported.

The kiosk reuses the backend's verification engine and model providers, which
read `backend.config.Settings` at import time. The kiosk must never pick up the
*server's* secrets or database from a `.env` file in the working directory, so
the relevant variables are set explicitly (environment variables take
precedence over `.env`). The kiosk issues no tokens and decrypts no server
templates, so these values are inert placeholders.
"""
from __future__ import annotations

import os
import secrets


def prepare(inference: str = "deepface") -> None:
    os.environ["SECRET_KEY"] = secrets.token_hex(32)
    os.environ["EMBEDDING_ENCRYPTION_KEY"] = ""
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    os.environ["AUTO_MIGRATE"] = "false"
    os.environ["INITIAL_ADMIN_PASSWORD"] = ""
    os.environ["SMTP_HOST"] = ""
    os.environ["INFERENCE_PROVIDER"] = "fake" if inference == "fake" else "deepface"
    # The fake provider is a test fixture and is refused by Settings outside APP_ENV=test.
    os.environ["APP_ENV"] = "test" if inference == "fake" else os.environ.get("KIOSK_APP_ENV", "production")
    os.environ.setdefault("LOG_LEVEL", "INFO")
