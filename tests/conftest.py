from typing import AsyncIterator

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from template.app import create_app


@pytest_asyncio.fixture(scope="session")
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


@pytest.fixture(scope="module")
async def engine():
    """Create a SQLite in-memory engine and initialize the schema."""
    from template.database import Base

    eng = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield eng
    await eng.dispose()


@pytest.fixture
async def session(engine) -> AsyncIterator[AsyncSession]:
    """Provide a transactional scope around a series of operations."""
    async_session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session
