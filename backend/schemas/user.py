"""User request/response schemas."""
from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime
from backend.models.user import RoleEnum


class UserCreate(BaseModel):
    email: str = Field(..., min_length=3)
    full_name: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=8)
    role: RoleEnum = RoleEnum.OPERATOR


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    role: Optional[RoleEnum] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    full_name: str
    role: RoleEnum
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None
    has_enrolment: bool = False


class UserListResponse(BaseModel):
    total: int
    users: list[UserResponse]
