from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,          # Set to True to print generated SQL statements in dev
    pool_pre_ping=True,  # Verifies connection health before handing it to a request
    pool_size=10,        # Max permanent connections in pool
    max_overflow=20,     # Max temporary connections under burst load
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Prevents attribute re-fetches after commits
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields a database session for a single HTTP request lifecycle."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()