"""Authentication and Role-Based Access Control (RBAC) dependencies (§8.1, §8.2)."""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any
import jwt
from fastapi import Cookie, Depends, HTTPException, Header, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from packages.cs_core.models import UserRole
from packages.cs_storage.db import get_sync_session
from packages.cs_storage.models_orm import UserAccountORM
from packages.cs_storage.repositories.user_repo import UserRepository

logger = logging.getLogger(__name__)

# Secret key and algorithm for cryptographically signed JWT tokens.
# If SECRET_KEY is not set in the environment, generate a random one at
# process startup rather than falling back to a hardcoded, production-looking
# default. This is stable for the lifetime of this process (so tests and a
# single running server behave consistently), but ephemeral: existing
# sessions/tokens are invalidated on every restart unless a real SECRET_KEY is
# configured. That tradeoff is intentional -- a hardcoded fallback secret is a
# far worse security posture than forcing re-login after a restart.
_env_secret = os.getenv("SECRET_KEY")
if _env_secret:
    SECRET_KEY = _env_secret
else:
    SECRET_KEY = secrets.token_hex(32)
    logger.warning(
        "[Auth] SECRET_KEY is not set in the environment! Generated a random, "
        "ephemeral secret key for this process. ALL existing JWT sessions will "
        "be invalidated on the next restart. Set the SECRET_KEY environment "
        "variable for a stable production deployment."
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours


class CurrentUser(BaseModel):
    user_id: int
    username: str
    role: UserRole


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a cryptographically signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(
    authorization: str | None = Header(None),
    session_token: str | None = Cookie(None),
) -> CurrentUser:
    """Resolve authenticated user by cryptographically validating the signed JWT token (§8.1)."""
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
    elif session_token:
        token = session_token

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Cryptographic JWT Signature Verification
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str = payload.get("sub")
        username = payload.get("username")
        role_str = payload.get("role")

        if user_id_str is None or username is None or role_str is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload structure.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        uid = int(user_id_str)
        role = UserRole(role_str)
        return CurrentUser(user_id=uid, username=username, role=role)

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (jwt.PyJWTError, ValueError, KeyError) as e:
        logger.warning(f"[Auth] Rejected unauthorized token verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid signature or malformed session token.",
            headers={"WWW-Authenticate": "Bearer"},
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
