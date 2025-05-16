from typing import AsyncIterator

import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import AsyncClient, ASGITransport

from template.app import create_app


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """
    Fixture to create an HTTP client for testing.

    Notes:
        See: https://github.com/Kludex/fastapi-tips?tab=readme-ov-file#5-use-httpxs-asyncclient-instead-of-testclient
    """
    app = create_app()

    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
