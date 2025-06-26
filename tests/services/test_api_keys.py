"""Unit-tests for the APIKeyService class.

These tests follow the same conventions used in the repository tests:
*   Async pytest style.
*   Google-style docstrings (English).
*   Concise, professional inline comments.
"""

from __future__ import annotations

import pytest
from fastapi.openapi.models import APIKey
from sqlalchemy.ext.asyncio import AsyncSession

from template.api.routes.schemas.api_keys import APIKeyCreate, APIKeyUpdate
from template.infrastructure.repositories.api_keys import APIKeyRepository
from template.application.api_keys import APIKeyService
from template.domain.api_keys import APIKeyNotFoundException, APIKeyNotProvidedException, APIKeyInvalidException


@pytest.fixture
def key_data() -> APIKeyCreate:
    """Static input data reused across several tests."""
    return APIKeyCreate(
        name="unit_test_service_key",
        description="Key created in service unit test",
        is_active=True,
    )


@pytest.fixture(scope="function")
async def repository(session: AsyncSession) -> APIKeyRepository:
    """Instantiate a repository bound to the provided DB session.

    Args:
        session (AsyncSession): Database session used for persistence.

    Returns:
        APIKeyRepository: Fresh repository instance for each test.
    """
    return APIKeyRepository(session)


@pytest.fixture(scope="function")
async def service(repository: APIKeyRepository) -> APIKeyService:
    """Return a ready-to-use service instance.

    Args:
        repository (APIKeyRepository): Repository fixture.

    Returns:
        APIKeyService: Service under test.
    """
    return APIKeyService(repository)


@pytest.fixture(scope="function")
async def created_key(service: APIKeyService, key_data: APIKeyCreate):
    """Create a key through the service and return the creation response."""
    return await service.create(key_data)


async def test_list_all_empty(service: APIKeyService):
    """list_all should return an empty collection when no keys exist.

    Args:
        service (APIKeyService): Service fixture.
    """
    keys = await service.list_all()
    assert keys == [], "Expected an empty list when no API keys are stored"


async def test_get_nonexistent_key_raises(service: APIKeyService):
    """get must raise APIKeyNotFoundException for an unknown ID."""
    with pytest.raises(APIKeyNotFoundException):
        await service.get(key_id=9999)


async def test_create_key_returns_plain_key(service: APIKeyService, key_data: APIKeyCreate):
    """create should return a response with both ID and one-time plain_key.

    Args:
        service (APIKeyService): Service fixture.
        key_data (APIKeyCreateSchema): Input schema for creation.
    """
    resp = await service.create(key_data)

    assert resp.id is not None, "Created key should have a generated ID"
    assert resp.plain_key, "plain_key must be returned on creation"
    assert resp.name == key_data.name
    assert resp.description == key_data.description
    assert resp.is_active is True


async def test_get_returns_same_key(service: APIKeyService, created_key: APIKey):
    """get should retrieve the same data that was created.

    Args:
        service (APIKeyService): Service fixture.
        created_key: Response object from previous creation.
    """
    fetched = await service.get(created_key.id)
    assert fetched.id == created_key.id
    assert fetched.name == created_key.name
    assert fetched.description == created_key.description
    assert fetched.is_active == created_key.is_active


async def test_list_all_after_creation(service: APIKeyService, created_key: APIKey):
    """list_all must include the newly created key."""
    keys = await service.list_all()
    assert len(keys) == 1, "Exactly one key expected after creation"
    assert keys[0].id == created_key.id


async def test_verify_key_valid(service: APIKeyService, created_key: APIKey):
    """verify_key should return True when a valid plain key is supplied."""
    assert await service.verify_key(created_key.plain_key) is True


async def test_verify_key_invalid(service: APIKeyService):
    """verify_key must raise APIKeyInvalidException for an unknown key."""
    with pytest.raises(APIKeyInvalidException):
        await service.verify_key("definitely-not-a-valid-key")


async def test_verify_key_not_provided(service: APIKeyService):
    """verify_key should raise when key header is missing."""
    with pytest.raises(APIKeyNotProvidedException):
        await service.verify_key(None)


async def test_update_key_description(service: APIKeyService, created_key: APIKey):
    """update must persist field changes and return updated schema."""
    new_desc = "Updated description via unit test"
    update_schema = APIKeyUpdate(
        name=created_key.name,
        description=new_desc,
        is_active=created_key.is_active,
    )

    updated = await service.update(created_key.id, update_schema)

    assert updated.description == new_desc, "Description should reflect update"


async def test_activate_and_deactivate_key(service: APIKeyService, created_key: APIKey):
    """activate helper should toggle the is_active flag."""
    # Deactivate
    inactive = await service.activate(created_key.id, active=False)
    assert not inactive.is_active, "Key should be inactive after deactivation"

    # Reactivate
    active = await service.activate(created_key.id, active=True)
    assert active.is_active, "Key should be active after reactivation"


async def test_delete_key(service: APIKeyService, created_key: APIKey):
    """delete should remove the key permanently and subsequent get should fail."""
    await service.delete(created_key.id)

    with pytest.raises(APIKeyNotFoundException):
        await service.get(created_key.id)

    remaining = await service.list_all()
    assert remaining == [], "Repository should be empty after deletion"
