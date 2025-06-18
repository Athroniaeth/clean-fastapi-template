from abc import ABC
from contextlib import asynccontextmanager
from typing import AsyncIterator, Type

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import DeclarativeBase


from sqlalchemy.ext.asyncio import async_sessionmaker


class Base(AsyncAttrs, DeclarativeBase):
    """Base to use for creating models."""


async def create_sessionmaker(database_url: str, base: Type[DeclarativeBase] = Base) -> async_sessionmaker:
    # Create an async engine
    engine = create_async_engine(
        url=database_url,
        future=True,
        pool_pre_ping=True,
    )

    # Create a session factory
    sessionmaker = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create the database schema
    async with engine.begin() as conn:
        await conn.run_sync(base.metadata.create_all)

    return sessionmaker


@asynccontextmanager
async def get_db(database_url: str, base: Type[DeclarativeBase] = Base) -> AsyncIterator[AsyncSession]:
    """Get the database session."""
    sessionmaker = await create_sessionmaker(database_url, base)

    async with sessionmaker() as async_session:
        try:
            yield async_session
        except Exception:
            await async_session.rollback()
            raise
        finally:
            await async_session.close()


class AbstractDatabaseInfra(ABC):
    """Abstract interface for file storage database."""

    _base: Type[DeclarativeBase]
    _engine: AsyncEngine
    _sessionmaker: async_sessionmaker

    def __init__(
        self,
        database_url: str,
        echo: bool = False,
        future: bool = True,
        expire_on_commit: bool = False,
        base: Type[DeclarativeBase] = Base,
    ):
        self._base = base
        self._engine = create_async_engine(database_url, echo=echo, future=future)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=expire_on_commit)
        self._session = self.session

    async def create_schema(self) -> None:
        """Create the database schema."""
        async with self._engine.begin() as conn:
            await conn.run_sync(self._base.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Context manager for database session."""
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                logger.error(f"Error during database operation: {e}")
                await session.rollback()
                raise
            finally:
                await session.close()


class PostgresDatabaseInfra(AbstractDatabaseInfra):
    """PostgreSQL database infrastructure using asyncpg."""

    def __init__(
        self,
        database_url: str = "postgresql+asyncpg://user:password@localhost/dbname",
        echo: bool = False,
        future: bool = True,
        expire_on_commit: bool = False,
    ):
        super().__init__(database_url, echo, future, expire_on_commit)


class InMemorySQLiteDatabaseInfra(AbstractDatabaseInfra):
    """SQLite in-memory database infrastructure using aiosqlite."""

    def __init__(
        self,
        database_url: str = "sqlite+aiosqlite:///:memory:",
        echo: bool = False,
        future: bool = True,
        expire_on_commit: bool = False,
    ):
        super().__init__(database_url, echo, future, expire_on_commit)
