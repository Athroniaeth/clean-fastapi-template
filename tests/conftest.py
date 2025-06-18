from functools import partial
from typing import AsyncIterator

import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient, ASGITransport
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession, AsyncEngine
from template.infrastructure.database.base import Base
from template.app import create_app, lifespan
from tests.test_settings import CustomSettings


@pytest.fixture(scope="session", autouse=True)
def cleanup_logger():
    """Pending execution of tests, disable the loguru logger."""
    logger.remove()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """
    Fixture to create an HTTP client for testing.

    Notes:
        See: https://github.com/Kludex/fastapi-tips?tab=readme-ov-file#5-use-httpxs-asyncclient-instead-of-testclient
    """
    settings = CustomSettings()

    # Delete lifespan (faster tests after finish running)
    app = create_app(
        title="Test App",
        version="0.1.0",
        description="Test Description",
        lifespan=partial(lifespan, settings=settings),
    )

    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
async def engine():
    """Create a SQLite in-memory engine and initialize the schema."""
    eng = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield eng
    await eng.dispose()


@pytest.fixture(scope="function")
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """
    Create a nested transaction and rollback at the end to isolate tests.
    """
    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with engine.connect() as conn:
        async with conn.begin():  # outer transaction
            session = async_session(bind=conn)

            # BEGIN SAVEPOINT
            await conn.execute(text("SAVEPOINT test_savepoint"))
            try:
                yield session
            finally:
                # ROLLBACK TO SAVEPOINT to clean up
                await conn.execute(text("ROLLBACK TO SAVEPOINT test_savepoint"))
                await session.close()
