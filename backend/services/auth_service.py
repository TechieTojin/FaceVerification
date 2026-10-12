"""Authentication service: password hashing, JWT tokens."""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models.user import User, RoleEnum

log = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except JWTError:
        return None


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, email: str, full_name: str, password: str, role: RoleEnum = RoleEnum.OPERATOR) -> User:
    user = User(
        email=email,
        full_name=full_name,
        hashed_password=hash_password(password),
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def seed_admin(db: Session) -> None:
    """Create default admin user if no users exist. Credentials come from settings/env."""
    from backend.config import settings
    if not settings.initial_admin_password:
        log.warning(
            "INITIAL_ADMIN_PASSWORD is not set in .env — skipping admin seed. "
            "Set it and restart to create the first admin account."
        )
        return
    count = db.query(User).count()
    if count == 0:
        admin = create_user(
            db,
            email=settings.initial_admin_email,
            full_name=settings.initial_admin_name,
            password=settings.initial_admin_password,
            role=RoleEnum.ADMIN,
        )
        log.warning("Default admin created: %s — CHANGE THIS PASSWORD IN PRODUCTION!", admin.email)
