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


def get_db_ssl_config() -> dict[str, Any]:
    """Inspect environment for mutual TLS / SSL database configuration (§4.4, §8.1)."""
    ssl_mode = os.getenv("DB_SSL_MODE")
    ssl_ca = os.getenv("DB_SSL_CA") or os.getenv("DB_SSL_ROOTCERT")
    ssl_cert = os.getenv("DB_SSL_CERT")
    ssl_key = os.getenv("DB_SSL_KEY")
    return {
        "ssl_mode": ssl_mode,
        "ssl_ca": ssl_ca,
        "ssl_cert": ssl_cert,
        "ssl_key": ssl_key,
        "is_mtls_configured": bool(ssl_ca and ssl_cert and ssl_key),
        "is_ssl_enabled": bool(ssl_mode or ssl_ca or ssl_cert),
    }


def get_sync_connect_args(target_url: str) -> dict[str, Any]:
    """Build connection arguments for sync SQLAlchemy engine (SQLite or PostgreSQL with mTLS)."""
    if "sqlite" in target_url:
        return {"check_same_thread": False, "timeout": 30}

    connect_args: dict[str, Any] = {}
    cfg = get_db_ssl_config()
    if cfg["ssl_mode"]:
        connect_args["sslmode"] = cfg["ssl_mode"]
    elif cfg["ssl_ca"]:
        connect_args["sslmode"] = "verify-ca"

    if cfg["ssl_ca"]:
        connect_args["sslrootcert"] = cfg["ssl_ca"]
    if cfg["ssl_cert"]:
        connect_args["sslcert"] = cfg["ssl_cert"]
    if cfg["ssl_key"]:
        connect_args["sslkey"] = cfg["ssl_key"]
    return connect_args


def get_async_connect_args(target_url: str) -> dict[str, Any]:
    """Build connection arguments for asyncpg engine with SSLContext supporting mTLS."""
    if "sqlite" in target_url:
        return {}

    cfg = get_db_ssl_config()
    if not cfg["is_ssl_enabled"]:
        return {}

    import ssl

    ssl_mode = (cfg["ssl_mode"] or ("verify-ca" if cfg["ssl_ca"] else "prefer")).lower()
    if ssl_mode in ("disable", "0", "false"):
        return {"ssl": False}

    ssl_ctx = ssl.create_default_context(
        cafile=cfg["ssl_ca"] if (cfg["ssl_ca"] and os.path.exists(cfg["ssl_ca"])) else None
    )
    if cfg["ssl_cert"] and cfg["ssl_key"]:
        if os.path.exists(cfg["ssl_cert"]) and os.path.exists(cfg["ssl_key"]):
            ssl_ctx.load_cert_chain(certfile=cfg["ssl_cert"], keyfile=cfg["ssl_key"])
        else:
            logger.warning(
                f"[Database mTLS] Certificate or key file not found: cert={cfg['ssl_cert']}, key={cfg['ssl_key']}"
            )

    if ssl_mode == "verify-full":
        ssl_ctx.check_hostname = True
        ssl_ctx.verify_mode = ssl.CERT_REQUIRED
    elif ssl_mode == "verify-ca":
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_REQUIRED
    elif ssl_mode in ("require", "prefer", "allow"):
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    return {"ssl": ssl_ctx}


def get_sync_engine(url: str | None = None) -> Any:
    global _sync_engine, _sync_session_factory
    if _sync_engine is None or url is not None:
        if _sync_engine is not None:
            _sync_engine.dispose()
        target_url = url or DEFAULT_SYNC_URL
        _check_db_environment(target_url)

        connect_args = get_sync_connect_args(target_url)
        if "sqlite" in target_url:
            _sync_engine = create_engine(target_url, echo=False, connect_args=connect_args)
        else:
            _sync_engine = create_engine(
                target_url,
                echo=False,
                pool_size=DB_POOL_SIZE,
                max_overflow=DB_MAX_OVERFLOW,
                pool_timeout=DB_POOL_TIMEOUT,
                pool_recycle=DB_POOL_RECYCLE,
                connect_args=connect_args,
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

        connect_args = get_async_connect_args(target_url)
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
                connect_args=connect_args,
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
