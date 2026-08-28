"""Repository for User accounts, authentication, and RBAC (§8.1)."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
import bcrypt
from sqlalchemy import select
from sqlalchemy.orm import Session
from packages.cs_storage.models_orm import UserAccountORM

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt with a unique random per-user salt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plaintext password against bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


class UserRepository:
    """Manages user authentication and role-based permissions."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_user(self, username: str, password: str, role: str) -> UserAccountORM:
        """Create a new user account with bcrypt hashed password."""
        user = UserAccountORM(
            username=username,
            role=role,
            hashed_password=hash_password(password),
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def update_password(self, username: str, new_password: str) -> bool:
        """Update password for an existing user."""
        user = self.get_by_username(username)
        if not user:
            return False
        user.hashed_password = hash_password(new_password)
        self.db.commit()
        return True

    def authenticate(self, username: str, password: str) -> UserAccountORM | None:
        """Authenticate user credentials using bcrypt."""
        user = self.get_by_username(username)
        if not user or not user.is_active:
            return None
        if verify_password(password, user.hashed_password):
            return user
        return None

    def get_by_username(self, username: str) -> UserAccountORM | None:
        """Fetch user by username."""
        return self.db.execute(
            select(UserAccountORM).where(UserAccountORM.username == username)
        ).scalar_one_or_none()

    def count_users(self) -> int:
        return self.db.query(UserAccountORM).count()

    def get_by_id(self, user_id: int) -> UserAccountORM | None:
        """Fetch user by ID."""
        return self.db.execute(
            select(UserAccountORM).where(UserAccountORM.id == user_id)
        ).scalar_one_or_none()

    def seed_default_users(self, defaults: list[tuple[str, str, str]] | None = None) -> None:
        """Seed initial accounts if database is empty."""
        if defaults is None:
            admin_pwd = os.getenv("ADMIN_DEFAULT_PASSWORD", "admin123")
            eng_pwd = os.getenv("ENGINEER_DEFAULT_PASSWORD", "eng123")
            op_pwd = os.getenv("OPERATOR_DEFAULT_PASSWORD", "op123")
            defaults = [
                ("admin", admin_pwd, "admin"),
                ("engineer", eng_pwd, "engineer"),
                ("operator", op_pwd, "operator"),
            ]
        for u, p, r in defaults:
            existing = self.get_by_username(u)
            if not existing:
                self.create_user(u, p, r)
            else:
                existing.hashed_password = hash_password(p)
                existing.role = r
                self.db.commit()
