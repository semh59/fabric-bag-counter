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
DEFAULT_ASYNC_URL = os.getenv("DATABASE_URL") or "sqlite+aiosqlite:///./data/cuval_sayim.db"
DEFAULT_SYNC_URL = os.getenv("DATABASE_SYNC_URL") or "sqlite:///./data/cuval_sayim.db"


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
        if _sync_engine is not None:
            # Replacing a live engine without disposing it would leak the old
            # engine's connection pool (open file handles / DB connections).
            _sync_engine.dispose()
        target_url = url or DEFAULT_SYNC_URL
        # check_same_thread=False: this app deliberately shares one engine/session
        # factory across threads (FastAPI's threadpool, the jobrunner/erp_relay
        # worker loops, concurrent test threads). Without a `timeout`, Python's
        # sqlite3 default busy_timeout is 0 -- a second thread's write while
        # another connection holds the write lock fails immediately with
        # "database is locked" instead of waiting, so any genuinely concurrent
        # write (e.g. two threads racing JobRepository.acquire_next_job) can
        # spuriously error rather than just serializing. 30s gives writers time
        # to queue up behind each other instead of raising.
        connect_args = {"check_same_thread": False, "timeout": 30} if "sqlite" in target_url else {}
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
        if _async_engine is not None:
            # See get_sync_engine: dispose the old engine before replacing it
            # so its pooled connections are actually released. AsyncEngine.dispose()
            # is a coroutine, but the underlying pool disposal itself is plain
            # sync work -- go through .sync_engine to dispose it without needing
            # an event loop here (this function is not async).
            _async_engine.sync_engine.dispose()
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
