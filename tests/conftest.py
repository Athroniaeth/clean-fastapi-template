from functools import partial
from typing import AsyncIterator

import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient, ASGITransport
from loguru import logger

from rename.infrastructure.database.adapter import InMemorySQLiteDatabaseInfra
from rename.infrastructure.database.base import AbstractDatabaseInfra, Base
from rename.infrastructure.storage.base import AbstractStorageInfra
from rename.infrastructure.storage.local import InMemoryStorageInfra


@pytest.fixture(scope="session", autouse=True)
def cleanup_logger():
    """Pending execution of tests, disable the loguru logger."""
    logger.remove()


@pytest.fixture(scope="session")
async def client() -> AsyncIterator[AsyncClient]:
    """
    Fixture to create an HTTP client for testing.

    Notes:
        See: https://github.com/Kludex/fastapi-tips?tab=readme-ov-file#5-use-httpxs-asyncclient-instead-of-testclient
    """
    from rename.api import create_app, lifespan
    from rename.settings import Settings

    settings = Settings()

    # Delete lifespan (faster tests after finish running)
    app = create_app(
        title="Test App",
        version="0.1.0",
        description="Test Description",
        lifespan=partial(lifespan, settings=settings),
    )

    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            yield client


@pytest.fixture(scope="session")
async def _infra_database() -> AsyncIterator[AbstractDatabaseInfra]:
    """Fixture to create an in-memory SQLite database for testing."""
    # Create SQLAlchemy models in the database
    infra_database = InMemorySQLiteDatabaseInfra(base=Base)
    await infra_database.create_schema()
    yield infra_database


@pytest.fixture(scope="function")
async def infra_database(_infra_database: AbstractDatabaseInfra) -> AsyncIterator[AbstractDatabaseInfra]:
    """Fixture to create an in-memory SQLite database for testing."""
    # Use the in-memory database infrastructure
    yield _infra_database

    # Clean up the database after the test
    await _infra_database.clean()


@pytest.fixture(scope="function")
async def infra_storage() -> AsyncIterator[AbstractStorageInfra]:
    """Fixture to create an in-memory storage infrastructure for testing."""
    # Create an in-memory storage infrastructure
    infra_storage = InMemoryStorageInfra()
    yield infra_storage
