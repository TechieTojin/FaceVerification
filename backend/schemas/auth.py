"""Auth request/response schemas — v2.1."""
from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class UserInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime


class LoginResponse(BaseModel):
    """Returned by POST /auth/login.

    The access_token is also delivered via HttpOnly cookie (fv_access).
    Programmatic clients may use the token from this body as a Bearer token.
    Browser clients should rely on the cookie.
    """
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserInfo


# Kept for backwards compatibility with existing test fixtures
TokenResponse = LoginResponse


class RefreshResponse(BaseModel):
    expires_in: int
    user: Optional[UserInfo] = None


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)
