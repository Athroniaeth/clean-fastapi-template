import time
from typing import AsyncIterator

import pytest

from template.application.api_keys import (
    ApiKeyService,
    APIKeyNotFoundException,
    APIKeyNotProvidedException,
    APIKeyInvalidException,
)
from template.domain.api_keys import ApiKey
from template.infrastructure.database.adapter import InMemorySQLiteDatabaseInfra
from template.infrastructure.repositories.api_keys import APIKeyRepository


ApiKey.generate_raw_key = lambda: "test_key_1234567890"  # Mock the key generation for testing purposes


@pytest.fixture(scope="function")
async def service(infra_database: InMemorySQLiteDatabaseInfra) -> AsyncIterator[ApiKeyService]:
    repo = APIKeyRepository(infra_database=infra_database)
    service = ApiKeyService(repo=repo)
    yield service


async def test_get_key(service: ApiKeyService):
    """Test creating and retrieving a api key."""
    time_start = time.time()
    # Create the api key
    api_key = await service.create(
        name="test_key",
        description="Test API Key",
        is_active=True,
    )
    print(f"Key created in {time.time() - time_start:.2f} seconds")

    # Retrieve the api key
    retrieved_key = await service.get(api_key.id_)
    print(f"Key retrieved in {time.time() - time_start:.2f} seconds")

    assert retrieved_key.id_ == api_key.id_, "Retrieved key ID does not match created key ID"
    assert retrieved_key.name == "test_key", "Retrieved key name does not match created key name"
    assert retrieved_key.description == "Test API Key", "Retrieved key description does not match created key description"
    assert retrieved_key.is_active is True, "Retrieved key is_active status does not match created key status"


async def test_create_key(service: ApiKeyService):
    """Test creating and deleting a api key."""

    # Create the api key
    api_key = await service.create(
        name="delete_test_key",
        description="Test API Key for Deletion",
        is_active=True,
    )

    # Delete the api key
    await service.delete(api_key)

    # Try to retrieve the deleted api key
    with pytest.raises(APIKeyNotFoundException):
        await service.get(api_key.id_)


async def test_update_key(service: ApiKeyService):
    """Test updating an existing api key."""

    # Create an api key to update
    api_key = await service.create(
        name="initial_key",
        description="Initial API Key",
        is_active=True,
    )
    # Retrieve the created key
    api_key = await service.get(api_key.id_)

    # Update the api key
    api_key.name = "Updated Key"
    api_key.description = "This is an updated API Key"
    api_key.is_active = False

    updated_key = await service.update(api_key)

    # Retrieve the updated key
    retrieved_key = await service.get(updated_key.id_)

    assert retrieved_key.name == "Updated Key", "Updated key name does not match"
    assert retrieved_key.description == "This is an updated API Key", "Updated key description does not match"
    assert retrieved_key.is_active is False, "Updated key is_active status does not match"


async def test_delete_key(service: ApiKeyService):
    """Test deleting an existing api key."""

    # Create an api key to delete
    api_key = ApiKey(
        name="not_persisted_key",
        description="This key will not be persisted",
        is_active=True,
        hashed_key="0" * 60,  # Mocked hashed key
        _plain_key="0" * 32,  # Mocked plain key
    )
    # Attempt to delete the key without persisting it
    with pytest.raises(APIKeyNotFoundException):
        await service.delete(api_key)


async def test_list_all_keys(service: ApiKeyService):
    """Test listing all api keys."""

    # Create multiple api keys
    await service.create(
        name="key_1",
        description="First API Key",
        is_active=True,
    )
    await service.create(
        name="key_2",
        description="Second API Key",
        is_active=False,
    )
    await service.create(
        name="key_3",
        description="Third API Key",
        is_active=True,
    )
    # List all api keys
    keys = await service.list_all()
    assert len(keys) >= 3, "Expected at least 3 keys to be listed"
    assert keys[0].name == "key_1", "First key name does not match expected"
    assert keys[1].name == "key_2", "Second key name does not match expected"
    assert keys[2].name == "key_3", "Third key name does not match expected"


async def test_list_active_keys(service: ApiKeyService):
    """Test listing only active api keys."""

    # Create multiple api keys with different active statuses
    await service.create(
        name="active_key_1",
        description="Active API Key 1",
        is_active=True,
    )
    await service.create(
        name="inactive_key_1",
        description="Inactive API Key 1",
        is_active=False,
    )
    await service.create(
        name="active_key_2",
        description="Active API Key 2",
        is_active=True,
    )

    # List only active api keys
    active_keys = await service.list_all(active_only=True)

    assert len(active_keys) == 2, "Expected exactly 2 active keys to be listed"
    assert all(key.is_active for key in active_keys), "All listed keys should be active"


async def test_get_non_existent_key(service: ApiKeyService):
    """Test retrieving a non-existent api key."""

    # Attempt to retrieve a key that does not exist
    with pytest.raises(APIKeyNotFoundException):
        await service.get(9999)  # Assuming 9999 is an ID that does not exist


async def test_get_key_without_providing_id(service: ApiKeyService):
    """Test retrieving a key without providing an ID."""

    # Attempt to retrieve a key without providing an ID
    with pytest.raises(APIKeyNotFoundException):
        # FastAPI return None if no key is provided
        await service.get(None)  # type: ignore


async def test_create_key_with_invalid_data(service: ApiKeyService):
    """Test creating a key with invalid data."""

    # Attempt to create a key with an empty name
    with pytest.raises(ValueError):
        await service.create(
            name="",  # Invalid name
            description="Invalid API Key",
            is_active=True,
        )

    # Attempt to create a key with a very long name
    with pytest.raises(ValueError):
        await service.create(
            name="x" * 65,  # Name exceeds maximum length
            description="Invalid API Key",
            is_active=True,
        )


async def test_create_key_with_long_description(service: ApiKeyService):
    """Test creating a key with a long description."""

    # Attempt to create a key with a very long description
    long_description = "x" * 300  # Exceeds maximum length of 256 characters
    with pytest.raises(ValueError):
        await service.create(
            name="long_description_key",
            description=long_description,
            is_active=True,
        )


async def test_create_key_with_special_characters(service: ApiKeyService):
    """Test creating a key with special characters in the name."""

    # Create a key with special characters in the name
    api_key = await service.create(
        name="special@key#1",
        description="API Key with Special Characters",
        is_active=True,
    )

    # Retrieve the key to verify it was created successfully
    retrieved_key = await service.get(api_key.id_)

    assert retrieved_key.name == "special@key#1", "Key name with special characters does not match"


async def test_update_non_existent_key(service: ApiKeyService):
    """Test updating a non-existent api key."""

    api_key = ApiKey(
        id_=9999,  # Assuming this ID does not exist
        name="Non-existent Key",
        description="This key does not exist",
        is_active=True,
        hashed_key="0" * 60,
        _plain_key="0" * 32,
    )

    # Attempt to update a key that does not exist
    with pytest.raises(APIKeyNotFoundException):
        await service.update(api_key)


async def test_delete_non_existent_key(service: ApiKeyService):
    """Test deleting a non-existent api key."""

    api_key = ApiKey(
        id_=9999,  # Assuming this ID does not exist
        name="Non-existent Key",
        description="This key does not exist",
        is_active=True,
        hashed_key="0" * 60,
        _plain_key="0" * 32,
    )
    # Attempt to delete a key that does not exist
    with pytest.raises(APIKeyNotFoundException):
        await service.delete(api_key)


async def test_activate_inactive_key(service: ApiKeyService):
    """Test activating an inactive api key."""

    # Create an inactive api key
    api_key = await service.create(
        name="inactive_key",
        description="Inactive API Key",
        is_active=False,
    )
    # Activate the key
    await service.activate(api_key.id_, True)

    api_key = await service.get(api_key.id_)
    assert api_key.is_active is True, "Key should be activated"


async def test_deactivate_active_key(service: ApiKeyService):
    """Test deactivating an active api key."""

    # Create an active api key
    api_key = await service.create(
        name="active_key",
        description="Active API Key",
        is_active=True,
    )
    # Deactivate the key
    deactivated_key = await service.activate(api_key.id_, False)
    api_key = await service.get(api_key.id_)
    assert deactivated_key.is_active is False, "Key should be deactivated"


async def test_verify_key(service: ApiKeyService):
    """Test verifying an API key."""

    # Create an API key
    api_key = await service.create(
        name="verify_key",
        description="API Key for Verification",
        is_active=True,
    )

    # Verify the key
    verified_key = await service.verify_key(api_key.plain_key)
    assert verified_key, "Key verification failed"


async def test_verify_invalid_key(service: ApiKeyService):
    """Test verifying an invalid API key."""

    # Attempt to verify an invalid key
    with pytest.raises(APIKeyInvalidException):
        await service.verify_key("invalid_key")  # Assuming "invalid_key" does not match any stored keys


async def test_verify_key_not_provided(service: ApiKeyService):
    """Test verifying an API key when no key is provided."""

    # Attempt to verify without providing a key
    with pytest.raises(APIKeyNotProvidedException):
        await service.verify_key(None)  # FastAPI return None if no key is provided
