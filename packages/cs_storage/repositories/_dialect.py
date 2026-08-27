"""Small shared helper for dialect-aware locking behaviour.

Row-level locking (`SELECT ... FOR UPDATE`, `SKIP LOCKED`) is a real-database
(Postgres) concept. SQLite -- used for local dev and the test suite -- has no
row-level locking at all (it locks at the database/connection level), and
raises a hard syntax error on `SKIP LOCKED`. Repositories that need atomic
claim/lock semantics check `is_postgres(db)` first so the same code path is
correct on Postgres in production and safely degrades on SQLite in tests.
"""

from __future__ import annotations

from sqlalchemy.orm import Session


def is_postgres(db: Session) -> bool:
    """True if this session is bound to a PostgreSQL engine."""
    bind = db.get_bind()
    return bool(bind is not None and bind.dialect.name == "postgresql")
