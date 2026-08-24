"""Authentication and Role-Based Access Control (RBAC) dependencies (§8.1, §8.2)."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Annotated, Any
from fastapi import Cookie, Depends, HTTPException, Header, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from packages.cs_core.models import UserRole
from packages.cs_storage.db import get_sync_session
from packages.cs_storage.models_orm import UserAccountORM
from packages.cs_storage.repositories.user_repo import UserRepository

# Secret key for signing / sessions
SECRET_KEY = os.getenv("SECRET_KEY", "cuval_secret_production_key_2026")


class CurrentUser(BaseModel):
    user_id: int
    username: str
    role: UserRole


def get_current_user(
    authorization: str | None = Header(None),
    session_token: str | None = Cookie(None),
) -> CurrentUser:
    """Resolve authenticated user from Authorization header or session cookie."""
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
    elif session_token:
        token = session_token

    if not token:
        # Default fallback for testing or unauthorized request
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials required.",
        )

    # Format: user_id:username:role:secret
    try:
        parts = token.split(":")
        if len(parts) >= 3:
            uid = int(parts[0])
            username = parts[1]
            role = UserRole(parts[2])
            return CurrentUser(user_id=uid, username=username, role=role)
    except Exception:
        pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired session token.",
    )


def require_role(*allowed_roles: UserRole):
    """RBAC dependency ensuring user has one of the allowed roles (§8.1)."""
    def role_checker(current_user: Annotated[CurrentUser, Depends(get_current_user)]) -> CurrentUser:
        if current_user.role == UserRole.ADMIN:
            # Admin has access to all routes
            return current_user
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied for role '{current_user.role.value}'. Requires: {[r.value for r in allowed_roles]}",
            )
        return current_user

    return role_checker
