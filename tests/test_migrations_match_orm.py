"""Guard against the migration file and the live ORM silently drifting apart.

Alembic is fully configured (alembic.ini, migrations/env.py,
migrations/versions/001_initial_schema.py) but nothing in the real
deployment ever actually invokes it -- every service bootstraps its schema
via packages.cs_storage.db.init_db_sync()'s Base.metadata.create_all(),
which only creates missing tables and never alters existing ones. That
combination is exactly how a real, previously-hidden bug reached a running
deployment this session: new columns were added to LineCalibrationORM and
to the migration file, but since the migration was never actually run
anywhere, the live Postgres database's line_calibration table didn't have
them -- discovered only via a real 500 error from the running API, fixed
with a manual ALTER TABLE against that one database.

This test runs `alembic upgrade head` for real (not create_all()) against
a throwaway SQLite database and checks its resulting schema has every
table/column the ORM declares, so the next time someone adds an ORM column
and forgets (or gets wrong) the matching migration edit, this test -- not
a live 500 error -- is what catches it.
"""

import os

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from packages.cs_storage.db import Base
from packages.cs_storage.models_orm import *  # noqa: F403  (populate Base.metadata)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_alembic_upgrade_head_runs_cleanly_on_sqlite(tmp_path):
    db_path = tmp_path / "migration_check.db"
    db_url = f"sqlite:///{db_path}"

    cfg = Config(os.path.join(REPO_ROOT, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(REPO_ROOT, "packages", "cs_storage", "migrations"))
    cfg.set_main_option("sqlalchemy.url", db_url)

    command.upgrade(cfg, "head")

    assert db_path.exists()


def test_migration_schema_has_every_orm_table_and_column(tmp_path):
    db_path = tmp_path / "migration_orm_check.db"
    db_url = f"sqlite:///{db_path}"

    cfg = Config(os.path.join(REPO_ROOT, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(REPO_ROOT, "packages", "cs_storage", "migrations"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")

    engine = create_engine(db_url)
    inspector = inspect(engine)
    migrated_tables = set(inspector.get_table_names())

    missing_tables = []
    missing_columns = {}
    for table_name, table in Base.metadata.tables.items():
        if table_name not in migrated_tables:
            missing_tables.append(table_name)
            continue
        migrated_columns = {c["name"] for c in inspector.get_columns(table_name)}
        orm_columns = {c.name for c in table.columns}
        missing = orm_columns - migrated_columns
        if missing:
            missing_columns[table_name] = missing

    engine.dispose()

    assert not missing_tables, f"ORM declares tables the migration never creates: {missing_tables}"
    assert not missing_columns, f"ORM declares columns the migration never creates: {missing_columns}"
