from logging.config import fileConfig
from sqlalchemy import create_engine, pool
from alembic import context

from packages.cs_storage.db import Base, DEFAULT_SYNC_URL
from packages.cs_storage.models_orm import *  # noqa: F403

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# alembic.ini's sqlalchemy.url is a placeholder -- override it with the same
# DATABASE_SYNC_URL/default resolution packages/cs_storage/db.py itself uses,
# so `alembic upgrade head` targets whatever database the real app actually
# connects to (Postgres in docker-compose, or the on-disk SQLite default)
# instead of silently migrating a URL nothing else in the app ever reads.
# Only when nothing else already overrode it, though -- a caller that set an
# explicit sqlalchemy.url before invoking (tests pointing at a throwaway
# database, `alembic -x` tooling) must win, not get silently clobbered back
# to the app default.
_PLACEHOLDER_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/bag_counter"
if config.get_main_option("sqlalchemy.url") in (None, "", _PLACEHOLDER_URL):
    config.set_main_option("sqlalchemy.url", DEFAULT_SYNC_URL)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # Plain sync engine, matching how the real app actually talks to the
    # database everywhere (get_sync_session/init_db_sync) -- the async
    # engine path in db.py has no real caller anywhere in services/ or
    # packages/, so using it here would need aiosqlite as a new dependency
    # purely to support this migration tool.
    connectable = create_engine(config.get_main_option("sqlalchemy.url"), poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
