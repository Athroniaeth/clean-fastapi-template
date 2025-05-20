import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from template.models.api_keys import ApiKeyModel
from template.repositories.api_keys import APIKeyRepository


@pytest.fixture
async def repository(session: AsyncSession) -> APIKeyRepository:
    """Return a ready-to-use repository instance.

    Args:
        session (AsyncSession): The database session.

    Returns:
        APIKeyRepository: An instance of the APIKeyRepository.
    """
    return APIKeyRepository(session)


@pytest.fixture(scope="module")
def new_key() -> ApiKeyModel:
    """Return a new API key instance for testing."""

    return ApiKeyModel(
        name="unit_test_key",
        description="Key created in unit test",
        is_active=True,
    )


async def test_list_all_empty(repository: APIKeyRepository):
    """Test that list_all returns an empty list when no API keys exist.

    Args:
        repository (APIKeyRepository): The API key repository fixture.
    """
    # At the start, repository should have no API keys
    api_keys = await repository.list_all()
    assert api_keys == [], "Expected no API keys in an empty repository"


async def test_create_and_get_api_key(repository: APIKeyRepository, new_key: ApiKeyModel):
    """Test creating an API key and retrieving it by ID.

    Args:
        repository (APIKeyRepository): The API key repository fixture.
        new_key (ApiKeyModel): The new API key instance.
    """
    # Create the key and verify an ID was assigned
    created = await repository.create(new_key)
    assert created.id is not None, "Created API key should have a non-null ID"

    # Retrieve the same key and compare fields
    fetched = await repository.get(created.id)
    assert fetched == created, "Fetched API key should match the created one"


async def test_list_all_after_creation(repository: APIKeyRepository, new_key: ApiKeyModel):
    """Test that list_all returns the newly created API key.

    Args:
        repository (APIKeyRepository): The API key repository fixture.
        new_key (ApiKeyModel): The new API key instance.
    """
    # list_all should now return a list containing our key
    all_keys = await repository.list_all()
    assert len(all_keys) == 1, "Expected exactly one API key after creation"
    assert all_keys[0].id == new_key.id, "The listed key should match the created key"


async def test_update_api_key_status(repository: APIKeyRepository, new_key: ApiKeyModel):
    """Test updating the is_active status of an API key.

    Args:
        repository (APIKeyRepository): The API key repository fixture.
        new_key (ApiKeyModel): The new API key instance.
    """
    # Toggle is_active to False
    await repository.update(new_key, {"is_active": False})

    # Fetch and ensure the status changed
    updated = await repository.get(new_key.id)
    assert updated is not None, "Updated API key should still exist"
    assert not updated.is_active, "API key should be inactive after update"


async def test_delete_api_key(repository: APIKeyRepository, new_key: ApiKeyModel):
    """Test deleting an API key removes it from the repository.

    Args:
        repository (APIKeyRepository): The API key repository fixture.
        new_key (ApiKeyModel): The new API key instance.
    """
    await repository.delete(new_key)

    # Attempt to fetch the deleted key
    deleted = await repository.get(new_key.id)
    assert deleted is None, "Deleted API key should not be retrievable"

    # Ensure list_all no longer includes it
    remaining = await repository.list_all()
    assert new_key not in remaining, "Deleted API key should not appear in list_all"
    assert remaining == [], "Repository should be empty after deletion"
