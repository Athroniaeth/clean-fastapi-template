from functools import lru_cache
from typing import AsyncIterator, Type

from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(AsyncAttrs, DeclarativeBase):
    """Base to use for creating models."""


@lru_cache(maxsize=3)
async def create_db(database_url: str, base: Type[DeclarativeBase] = Base) -> sessionmaker:
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

    # Create the database schema
    async with engine.begin() as conn:
        await conn.run_sync(base.metadata.create_all)

    return async_session


async def get_db(database_url: str, base: Type[DeclarativeBase] = Base) -> AsyncIterator[AsyncSession]:
    """Get the database session."""
    async_session = await create_db(database_url, base)

    async with async_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
