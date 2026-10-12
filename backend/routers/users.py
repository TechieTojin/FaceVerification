"""User management router."""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from backend.database import get_db
from backend.models.user import User, RoleEnum
from backend.schemas.user import UserCreate, UserUpdate, UserResponse, UserListResponse
from backend.services.auth_service import create_user, hash_password, get_user_by_email
from backend.routers.auth import get_current_user, require_admin

router = APIRouter(prefix="/api/v1/users", tags=["users"])


def _user_to_response(user: User, db: Session) -> UserResponse:
    from backend.models.biometric import BiometricEnrolment
    enrolment = db.query(BiometricEnrolment).filter(BiometricEnrolment.user_id == user.id).first()
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login_at=user.last_login_at,
        has_enrolment=enrolment is not None and enrolment.enrolled_at is not None,
    )


@router.get("", response_model=UserListResponse)
def list_users(
    skip: int = 0,
    limit: int = 50,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    q = db.query(User)
    if search:
        q = q.filter(User.email.contains(search) | User.full_name.contains(search))
    total = q.count()
    users = q.offset(skip).limit(limit).all()
    return UserListResponse(
        total=total,
        users=[_user_to_response(u, db) for u in users],
    )


@router.post("", response_model=UserResponse, status_code=201)
def create_new_user(
    data: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    existing = get_user_by_email(db, data.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = create_user(db, data.email, data.full_name, data.password, data.role)
    return _user_to_response(user, db)


@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return _user_to_response(user, db)


@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: str,
    data: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if data.full_name is not None:
        user.full_name = data.full_name
    if data.role is not None:
        user.role = data.role
    if data.is_active is not None:
        user.is_active = data.is_active
    db.commit()
    db.refresh(user)
    return _user_to_response(user, db)


@router.delete("/{user_id}", status_code=204)
def delete_user(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    db.delete(user)
    db.commit()


@router.post("/{user_id}/disable", response_model=UserResponse)
def disable_user(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = False
    db.commit()
    db.refresh(user)
    return _user_to_response(user, db)
