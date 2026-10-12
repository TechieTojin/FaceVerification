"""System settings router — v2.1 with floor/ceiling validation."""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
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

# (min, max) inclusive ranges — enforce floor to prevent security misconfig
_FLOAT_BOUNDS: dict[str, tuple[float, float]] = {
    "max_face_distance": (0.1, 1.0),
    "min_face_quality": (0.1, 1.0),
}
_INT_BOUNDS: dict[str, tuple[int, int]] = {
    "session_timeout_seconds": (30, 3600),
    "max_attempts_per_session": (3, 50),
}
_BOOL_KEYS = {"enable_liveness"}


def _validate_setting(key: str, value: str) -> str:
    """Validate and normalise a setting value. Raises HTTPException on bad input."""
    if key in _BOOL_KEYS:
        if value.lower() not in ("true", "false", "1", "0"):
            raise HTTPException(status_code=422, detail=f"{key}: must be 'true' or 'false'")
        return "true" if value.lower() in ("true", "1") else "false"

    if key in _FLOAT_BOUNDS:
        try:
            v = float(value)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"{key}: must be a number")
        lo, hi = _FLOAT_BOUNDS[key]
        if not (lo <= v <= hi):
            raise HTTPException(status_code=422, detail=f"{key}: must be between {lo} and {hi}")
        return str(v)

    if key in _INT_BOUNDS:
        try:
            v = int(value)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"{key}: must be an integer")
        lo, hi = _INT_BOUNDS[key]
        if not (lo <= v <= hi):
            raise HTTPException(status_code=422, detail=f"{key}: must be between {lo} and {hi}")
        return str(v)

    return str(value)


def _ensure_defaults(db: Session) -> None:
    for key, (value, description) in DEFAULT_SETTINGS.items():
        if not db.query(SystemSettings).filter(SystemSettings.key == key).first():
            db.add(SystemSettings(key=key, value=value, description=description))
    db.commit()


@router.get("")
def get_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _ensure_defaults(db)
    rows = db.query(SystemSettings).all()
    return {
        s.key: {
            "value": s.value,
            "description": s.description,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
        }
        for s in rows
    }


@router.put("")
def update_settings(
    data: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    _ensure_defaults(db)
    unknown = [k for k in data if k not in DEFAULT_SETTINGS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown settings keys: {unknown}")

    updated = {}
    for key, raw_value in data.items():
        validated = _validate_setting(key, str(raw_value))
        setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()
        if setting:
            setting.value = validated
            setting.updated_by = current_user.id
            setting.updated_at = datetime.utcnow()
            updated[key] = validated
    db.commit()
    return {"updated": updated, "message": f"Updated {len(updated)} settings"}
