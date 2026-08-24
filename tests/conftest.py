"""Pytest global fixtures for resetting database state between tests."""

import pytest
from packages.cs_storage.db import Base, get_sync_engine, init_db_sync


@pytest.fixture(autouse=True)
def clean_db():
    """Reset database tables before each test."""
    engine = get_sync_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
