from httpx import AsyncClient


async def test_root(client: AsyncClient) -> None:
    """Test the root endpoint."""
    response = await client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "name": "Test App",
        "version": "0.1.0",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


async def test_health(client: AsyncClient) -> None:
    """Test the health endpoint."""
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
