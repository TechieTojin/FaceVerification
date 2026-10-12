"""System settings router."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime

from backend.database import get_db
from backend.models.settings import SystemSettings
from backend.models.user import User
from backend.routers.auth import get_current_user, require_admin

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

DEFAULT_SETTINGS = {
    "max_face_distance": ("0.68", "Maximum cosine distance for identity match"),
    "min_face_quality": ("0.6", "Minimum quality score for enrolment"),
    "enable_liveness": ("true", "Enable liveness/anti-spoofing check"),
    "session_timeout_seconds": ("120", "Verification session timeout in seconds"),
    "max_attempts_per_session": ("5", "Maximum frame submissions per session"),
}


def _ensure_defaults(db: Session):
    for key, (value, description) in DEFAULT_SETTINGS.items():
        existing = db.query(SystemSettings).filter(SystemSettings.key == key).first()
        if not existing:
            db.add(SystemSettings(key=key, value=value, description=description))
    db.commit()


@router.get("")
def get_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _ensure_defaults(db)
    settings = db.query(SystemSettings).all()
    return {s.key: {"value": s.value, "description": s.description, "updated_at": s.updated_at.isoformat() if s.updated_at else None} for s in settings}


@router.put("")
def update_settings(
    data: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    _ensure_defaults(db)
    updated = {}
    for key, value in data.items():
        if key in DEFAULT_SETTINGS:
            setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()
            if setting:
                setting.value = str(value)
                setting.updated_by = current_user.id
                setting.updated_at = datetime.utcnow()
                updated[key] = str(value)
    db.commit()
    return {"updated": updated, "message": f"Updated {len(updated)} settings"}
