from typing import Sequence, Optional

from starlette import status

from template.api.core.exceptions import APIException
from template.domain.api_keys import ApiKey
from template.infrastructure.repositories.api_keys import APIKeyRepository


class APIKeyException(APIException):
    """Base class for API key-related exceptions."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)
        self.status_code = status_code
        self.detail = detail


class APIKeyNotFoundException(APIKeyException):
    """Raised when an API key is not found in the database."""

    status_code = status.HTTP_404_NOT_FOUND
    detail = "API key not found : {key_id}"

    def __init__(self, key_id: int):
        super().__init__(
            status_code=self.status_code,
            detail=self.detail.format(key_id=key_id),
        )


class APIKeyNotProvidedException(APIKeyException):
    """Raised when an API key is not provided in the request."""

    status_code = status.HTTP_401_UNAUTHORIZED
    detail = "API key not provided in the request."

    def __init__(self):
        super().__init__(
            status_code=self.status_code,
            detail=self.detail,
        )


class APIKeyInvalidException(APIKeyException):
    """Raised when an API key is invalid or does not match any stored keys."""

    status_code = status.HTTP_401_UNAUTHORIZED
    detail = "Invalid API key provided : {raw_key}"

    def __init__(self, raw_key: str):
        super().__init__(
            status_code=self.status_code,
            detail=self.detail.format(raw_key=raw_key),
        )


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
        api_key = ApiKey.create(
            name=name,
            description=description,
            is_active=is_active,
        )

        # Persist the model
        return await self._repo.create(api_key)

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

        # Todo : Create a custom repo_meta query for this (return boolean)
        # And this method will check the boolean, and raise an exception if False
        for key in keys:
            if key.check_key(raw_key):
                return True

        raise APIKeyInvalidException(raw_key)
