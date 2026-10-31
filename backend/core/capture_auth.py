"""Who may drive a capture session's frames.

Besides signed-in operators, two non-session principals capture frames:
  * `X-Login-Token`  — a pending passkey sign-in (or face step-up) doing its live face check;
  * `X-Pairing-Secret` — a phone/browser that claimed a QR pairing or hosted verification link.
Both are header credentials (never cookies), so they are not exposed to CSRF.
"""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from backend.core.deps import Actor, get_actor, oauth2_scheme
from backend.database import get_db


def get_capture_actor(request: Request, db: Session = Depends(get_db),
                      bearer: Optional[str] = Depends(oauth2_scheme)) -> Actor:
    login_token = request.headers.get("X-Login-Token")
    if login_token:
        from backend.services import mfa_service
        return mfa_service.face_actor(db, login_token)
    pairing_secret = request.headers.get("X-Pairing-Secret")
    if pairing_secret:
        from backend.services import pairing_service
        return pairing_service.actor_for_secret(db, pairing_secret)
    return get_actor(request, db, bearer)
