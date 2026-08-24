"""Database engine initialization, session management, and Base declaration."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Generator
from typing import Any
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Database URLs from environment or sensible defaults (SQLite for tests/local, Postgres for prod)
DEFAULT_ASYNC_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./data/cuval_sayim.db" if "sqlite" in os.getenv("DATABASE_URL", "") else "sqlite+aiosqlite:///./test_cuval.db"
)
DEFAULT_SYNC_URL = os.getenv(
    "DATABASE_SYNC_URL",
    "sqlite:///./data/cuval_sayim.db" if "sqlite" in os.getenv("DATABASE_SYNC_URL", "") else "sqlite:///./test_cuval.db"
)


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy ORM models."""
    pass


# Global engine and session makers
_async_engine = None
_sync_engine = None
_async_session_factory = None
_sync_session_factory = None


def get_sync_engine(url: str | None = None) -> Any:
    global _sync_engine, _sync_session_factory
    if _sync_engine is None or url is not None:
        target_url = url or DEFAULT_SYNC_URL
        connect_args = {"check_same_thread": False} if "sqlite" in target_url else {}
        _sync_engine = create_engine(target_url, echo=False, connect_args=connect_args)
        _sync_session_factory = sessionmaker(bind=_sync_engine, expire_on_commit=False)
    return _sync_engine


def get_sync_session(url: str | None = None) -> Session:
    global _sync_session_factory
    if _sync_session_factory is None or url is not None:
        get_sync_engine(url)
    assert _sync_session_factory is not None
    return _sync_session_factory()


def get_async_engine(url: str | None = None) -> Any:
    global _async_engine, _async_session_factory
    if _async_engine is None or url is not None:
        target_url = url or DEFAULT_ASYNC_URL
        _async_engine = create_async_engine(target_url, echo=False)
        _async_session_factory = async_sessionmaker(bind=_async_engine, expire_on_commit=False, class_=AsyncSession)
    return _async_engine


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for yielding async db sessions."""
    global _async_session_factory
    if _async_session_factory is None:
        get_async_engine()
    assert _async_session_factory is not None
    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def init_db_sync(url: str | None = None) -> None:
    """Create all tables synchronously (useful for bootstrap & unit tests)."""
    engine = get_sync_engine(url)
    Base.metadata.create_all(bind=engine)


async def init_db(url: str | None = None) -> None:
    """Create all tables asynchronously."""
    engine = get_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
