import time
from typing import Sequence, Optional

from template.domain.api_keys import APIKeyNotFoundException, APIKeyNotProvidedException, APIKeyInvalidException, ApiKey
from template.infrastructure.repositories.api_keys import APIKeyRepository


class ApiKeyService:
    """
    Service layer for managing API keys.

    Attributes:
        _repo (APIKeyRepository): repository for API key operations.
    """

    def __init__(self, repo: APIKeyRepository):
        self._repo = repo

    async def _get_key(self, key_id: int) -> ApiKey:
        """
        Helper method to retrieve an API key by its ID.

        Args:
            key_id (int): identifier of the key.

        Returns:
            ApiKey: the retrieved key.
        """
        key = await self._repo.get_by_id(key_id)

        if not key:
            raise APIKeyNotFoundException(key_id)

        return key

    async def get(self, id_: int) -> ApiKey:
        """
        Retrieve an API key by its ID.

        Args:
            id_ (int): identifier of the key.

        Returns:
            APIKeyRead: the retrieved key.

        Raises:
            APIKeyNotFoundException: if no such key exists.
        """
        return await self._get_key(id_)

    async def list_all(self, skip: int = 0, limit: int = 100, active_only: bool = False) -> Sequence[ApiKey]:
        """
        List API keys with optional pagination and active‐only filtering.

        Args:
            skip (int): number of records to skip.
            limit (int): maximum number to return.
            active_only (bool): if True, only return active keys.

        Returns:
            List[ApiKey]: list of key schemas.
        """
        return await self._repo.list_all(
            skip=skip,
            limit=limit,
            active_only=active_only,
        )

    async def create(
        self,
        name: str,
        description: Optional[str] = None,
        is_active: bool = True,
    ) -> ApiKey:
        """
        Create and persist a new API key.

        Args:
            name (str): human-readable name for the key.
            description (Optional[str]): optional description.
            is_active (bool): whether the key should be active upon creation.

        Returns:
            ApiKey: the created key + its raw plain_key.
        """
        time_elapsed = time.time()
        api_key = ApiKey.create(
            name=name,
            description=description,
            is_active=is_active,
        )
        print(f"\tKey created (pydantic) in {time.time() - time_elapsed:.2f} seconds")
        # Persist the model
        api_key = await self._repo.create(api_key)
        print(f"\tKey persisted in {time.time() - time_elapsed:.2f} seconds")
        return api_key

    async def update(self, api_key: ApiKey) -> ApiKey:
        """
        Update fields of an existing API key.

        Args:
            api_key (APIKeyUpdate): fields to modify.

        Returns:
            ApiKey: updated key.

        Raises:
            APIKeyNotFoundException: if no such key exists.
        """
        if not await self._repo.update(api_key):
            raise APIKeyNotFoundException(api_key.id_)

        return api_key

    async def delete(self, api_key: ApiKey) -> None:
        """
        Permanently delete an API key by its ID.

        Args:
            api_key (ApiKey): the key

        Raises:
            APIKeyNotFoundException: if no such key exists.
        """
        if not await self._repo.delete(api_key):
            raise APIKeyNotFoundException(api_key.id_)

    async def activate(self, key_id: int, active: bool) -> ApiKey:
        """
        Activate or deactivate an API key.

        Args:
            key_id (int): identifier of the key.
            active (bool): True to activate, False to deactivate.

        Returns:
            APIKeyRead: the updated key.

        Raises:
            APIKeyNotFoundException: if no such key exists.
        """
        key = await self._get_key(key_id)
        key.is_active = active
        await self._repo.update(key)
        return key

    async def verify_key(self, raw_key: Optional[str]) -> bool:
        """
        Verify that a raw API key is valid against stored hashes.

        Only active keys are checked.

        Args:
            raw_key (str): the plain key to verify.

        Returns:
            bool: True if valid.

        Raises:
            APIKeyInvalidException: if no match is found.
        """
        if not raw_key:
            raise APIKeyNotProvidedException()

        keys = await self._repo.list_all(active_only=True)

        # Todo : Create a custom repo query for this (return boolean)
        # And this method will check the boolean, and raise an exception if False
        for key in keys:
            if key.check_key(raw_key):
                return True

        raise APIKeyInvalidException(raw_key)
