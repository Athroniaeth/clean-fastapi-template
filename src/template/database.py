from typing import AsyncIterator

from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase

"""url = URL.create(
    drivername="postgresql",
    username="postgres",
    password="",
    host="localhost",
    database="mydb",
    port=5432
)"""

url = URL.create(
    drivername="sqlite+aiosqlite",
    database="./data/db.sqlite",
)

# Create an async engine
engine = create_async_engine(
    url=url,
    future=True,
    pool_pre_ping=True,
)

# Create a session factory
async_session = sessionmaker(  # noqa
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(AsyncAttrs, DeclarativeBase):
    """Base to use for creating models."""


async def get_db() -> AsyncIterator[AsyncSession]:
    async with async_session() as session:
        yield session


async def create_database() -> None:
    """Create the database from the models."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
