"""Repository for User accounts, authentication, and RBAC (§8.1)."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Sequence
from sqlalchemy import select
from sqlalchemy.orm import Session
from packages.cs_storage.models_orm import UserAccountORM


def _hash_password(password: str) -> str:
    salt = os.getenv("AUTH_SALT", "cuval_salt_2026")
    return hashlib.sha256(f"{password}:{salt}".encode("utf-8")).hexdigest()


class UserRepository:
    """Manages user authentication and role-based permissions."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_user(self, username: str, password: str, role: str) -> UserAccountORM:
        """Create a new user account with hashed password."""
        user = UserAccountORM(
            username=username,
            role=role,
            hashed_password=_hash_password(password),
            is_active=True,
            created_at=datetime.utcnow(),
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def authenticate(self, username: str, password: str) -> UserAccountORM | None:
        """Authenticate user credentials."""
        user = self.get_by_username(username)
        if not user or not user.is_active:
            return None
        if user.hashed_password == _hash_password(password):
            return user
        return None

    def get_by_username(self, username: str) -> UserAccountORM | None:
        """Fetch user by username."""
        return self.db.execute(
            select(UserAccountORM).where(UserAccountORM.username == username)
        ).scalar_one_or_none()

    def get_by_id(self, user_id: int) -> UserAccountORM | None:
        """Fetch user by ID."""
        return self.db.execute(
            select(UserAccountORM).where(UserAccountORM.id == user_id)
        ).scalar_one_or_none()

    def seed_default_users(self) -> None:
        """Seed initial default accounts if database is empty."""
        defaults = [
            ("admin", "admin123", "admin"),
            ("engineer", "eng123", "engineer"),
            ("operator", "op123", "operator"),
        ]
        for u, p, r in defaults:
            if not self.get_by_username(u):
                self.create_user(u, p, r)
