from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from core.config import settings
from db.models import Base

config = context.config
fileConfig(config.config_file_name)

# point Alembic at your models so it can detect schema changes
target_metadata = Base.metadata


def get_url() -> str:
    # use the local URL (localhost) since Alembic runs on your machine, not in Docker
    url = settings.postgres_url_local
    if not url:
        raise ValueError(
            "POSTGRES_URL_LOCAL is not set in .env — "
            "add it as postgresql://scholar:changeme@localhost:5432/scholarflow"
        )
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,   # don't hold connections open between migrations
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()