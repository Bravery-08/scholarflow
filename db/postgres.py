from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import create_engine
from core.config import settings
from db.models import Base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager


def _make_async_url(base_url: str) -> str:
    return base_url.replace("postgresql://", "postgresql+asyncpg://")


# ── Docker async engine (used by FastAPI containers) ─────────────────────────
async_engine = create_async_engine(
    _make_async_url(settings.postgres_url),
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

# ── Local async engine (used by scripts running on your machine) ──────────────
_local_url = settings.postgres_url_local or settings.postgres_url
local_async_engine = create_async_engine(
    _make_async_url(_local_url),
    echo=settings.debug,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

LocalAsyncSessionLocal = async_sessionmaker(
    bind=local_async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    """FastAPI dependency — uses the Docker engine."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_local_db() -> AsyncSession:
    """For scripts running on your local machine — uses localhost."""
    async with LocalAsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    async with local_async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def get_sync_engine():
    return create_engine(settings.postgres_url_local, echo=False)

# ── Sync sessions (for Dramatiq workers and local scripts) ───────────────────

def _make_sync_engine(url: str):
    return create_engine(url, echo=settings.debug, pool_pre_ping=True)

@contextmanager
def get_sync_session(local: bool = False):
    """
    Context manager yielding a sync SQLAlchemy session.
    local=True  → uses localhost (scripts on your machine)
    local=False → uses Docker service name (Dramatiq workers in containers)
    """
    url = settings.postgres_url_local if local else settings.postgres_url
    engine = _make_sync_engine(url)
    SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()