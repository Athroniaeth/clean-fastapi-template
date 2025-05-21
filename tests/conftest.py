import asyncio
from typing import AsyncIterator

import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession, AsyncEngine
from template.database import Base
from template.app import create_app


@pytest.fixture(scope="session")
def event_loop():
    """
    asyncio_default_fixture_loop_scope pyproject.toml variable
    define all fixtures with 'function' scope, we need to
    override it to 'package' scope.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def client() -> AsyncIterator[AsyncClient]:
    """
    Fixture to create an HTTP client for testing.

    Notes:
        See: https://github.com/Kludex/fastapi-tips?tab=readme-ov-file#5-use-httpxs-asyncclient-instead-of-testclient
    """
    # Delete lifespan (faster tests after finish running)
    app = create_app()

    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture(scope="session")
async def engine():
    """Create a SQLite in-memory engine and initialize the schema."""
    eng = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield eng
    await eng.dispose()


@pytest.fixture
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
