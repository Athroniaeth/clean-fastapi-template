from functools import lru_cache
from typing import AsyncIterator, Type

from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncEngine, create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(AsyncAttrs, DeclarativeBase):
    """Base to use for creating models."""


@lru_cache(maxsize=3)
async def get_db(database_url: str, base: Type[DeclarativeBase] = Base) -> (AsyncSession, AsyncEngine):
    # Create an async engine
    engine = create_async_engine(
        url=database_url,
        future=True,
        pool_pre_ping=True,
    )

    # Create a session factory
    async_session = sessionmaker(  # noqa
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    await create_database(engine, base)
    return async_session


async def _inject_db(database_url: str, base: Type[DeclarativeBase] = Base) -> AsyncIterator[AsyncSession]:
    """Get the database session."""
    async_session = get_db(database_url, base)

    async with async_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_database(engine: AsyncEngine, base: Type[DeclarativeBase]) -> None:
    """Create the database from the models."""
    async with engine.begin() as conn:
        await conn.run_sync(base.metadata.create_all)
