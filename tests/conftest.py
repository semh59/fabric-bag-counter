"""Pytest global fixtures for resetting database state between tests."""

import os

# Point the storage layer at an isolated, per-test-process SQLite file BEFORE
# packages.cs_storage.db is imported -- its DEFAULT_SYNC_URL / DEFAULT_ASYNC_URL
# are computed once at import time from these same env vars. Without this,
# tests fall back to db.py's real dev/prod default (./data/cuval_sayim.db),
# the same file a manually-run `uvicorn services.api.main:app` (or any other
# process working in this checkout) would use. The autouse `clean_db` fixture
# below does a DROP-then-CREATE of every table before *every single test*, so
# sharing that file with anything else running concurrently causes real
# damage: a table briefly not existing mid-DROP/CREATE from this process gets
# read by the other one ("no such table: ..."), or two processes each believe
# they just emptied a table and both try to insert the same seed rows into it
# ("UNIQUE constraint failed: ..."). A PID-suffixed file guarantees this test
# run never shares a database with anything else, no matter what else is
# running against this repo checkout at the same time. (Respects an
# explicitly-set DATABASE_SYNC_URL/DATABASE_URL via setdefault -- e.g.
# pointing tests at a real Postgres instance to exercise that dialect --
# rather than overriding it.)
_TEST_DB_PATH = f"./data/test_cuval_{os.getpid()}.db"
os.environ.setdefault("DATABASE_SYNC_URL", f"sqlite:///{_TEST_DB_PATH}")
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_TEST_DB_PATH}")

import pytest
from packages.cs_storage.db import Base, get_sync_engine, init_db_sync


@pytest.fixture(autouse=True)
def clean_db():
    """Reset database tables before each test.

    The production login endpoint no longer auto-seeds default accounts
    (that was demo-flavored behavior removed from the real app -- see
    main.py's real first-run bootstrap instead). Test files that log in
    with the fixed operator/op123, engineer/eng123, admin/admin123 accounts
    are responsible for seeding those users themselves (most already do, via
    UserRepository.seed_default_users() or create_user() in their own
    setup); intentionally not done globally here, since some test files
    create those exact usernames directly and a blanket seed here would
    collide with those.
    """
    engine = get_sync_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture(scope="session", autouse=True)
def _cleanup_test_db_file():
    """Remove this test session's private SQLite file when the run finishes."""
    yield
    try:
        engine = get_sync_engine()
        engine.dispose()
    except Exception:
        pass
    path = _TEST_DB_PATH[2:] if _TEST_DB_PATH.startswith("./") else _TEST_DB_PATH
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass  # best-effort cleanup; a lingering per-PID file is harmless
