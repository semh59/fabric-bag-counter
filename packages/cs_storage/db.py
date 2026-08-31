"""Database engine initialization, session management, and Base declaration."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

# Database URLs from environment or sensible defaults.
#
# When DATABASE_URL / DATABASE_SYNC_URL are set (e.g. via docker-compose for
# Postgres), they are used as-is. When unset, we default to a real, clearly
# named on-disk SQLite database under ./data -- NOT a "test" database. Falling
# back to a file named test_cuval.db by default meant a real (non-test) run
# with no env configured would silently write to what looks like throwaway
# test data. Tests get their own isolated state via conftest.py's per-test
# drop_all/create_all, so sharing this default path with a real run is safe.
DEFAULT_ASYNC_URL = os.getenv("DATABASE_URL") or "sqlite+aiosqlite:///./data/bag_counter.db"
DEFAULT_SYNC_URL = os.getenv("DATABASE_SYNC_URL") or "sqlite:///./data/bag_counter.db"


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy ORM models."""
    pass


# Global engine and session makers
_async_engine = None
_sync_engine = None
_async_session_factory = None
_sync_session_factory = None


# Connection pool sizing for PostgreSQL / production databases
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))

ENV_MODE = os.getenv("ENV", os.getenv("NODE_ENV", "development")).lower()


def _check_db_environment(target_url: str) -> None:
    if "sqlite" in target_url and ENV_MODE in ["production", "prod"]:
        logger.warning(
            "[Database] WARNING: Running on SQLite in PRODUCTION environment! "
            "For multi-worker and high-throughput concurrent deployments, configure PostgreSQL via DATABASE_URL."
        )


def get_sync_engine(url: str | None = None) -> Any:
    global _sync_engine, _sync_session_factory
    if _sync_engine is None or url is not None:
        if _sync_engine is not None:
            _sync_engine.dispose()
        target_url = url or DEFAULT_SYNC_URL
        _check_db_environment(target_url)

        if "sqlite" in target_url:
            connect_args = {"check_same_thread": False, "timeout": 30}
            _sync_engine = create_engine(target_url, echo=False, connect_args=connect_args)
        else:
            _sync_engine = create_engine(
                target_url,
                echo=False,
                pool_size=DB_POOL_SIZE,
                max_overflow=DB_MAX_OVERFLOW,
                pool_timeout=DB_POOL_TIMEOUT,
                pool_recycle=DB_POOL_RECYCLE,
            )
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
        if _async_engine is not None:
            _async_engine.sync_engine.dispose()
        target_url = url or DEFAULT_ASYNC_URL
        _check_db_environment(target_url)
        if "sqlite" in target_url:
            _async_engine = create_async_engine(target_url, echo=False)
        else:
            _async_engine = create_async_engine(
                target_url,
                echo=False,
                pool_size=DB_POOL_SIZE,
                max_overflow=DB_MAX_OVERFLOW,
                pool_timeout=DB_POOL_TIMEOUT,
                pool_recycle=DB_POOL_RECYCLE,
            )
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
            logger.exception("Database session failed; rolling back transaction.")
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
