from typing import Sequence, Optional

from fastapi.security import APIKeyHeader
from starlette import status

from template.core.exceptions import APIException
from template.models.api_keys import ApiKeyModel
from template.repositories.api_keys import APIKeyRepository
from template.schemas.api_keys import (
    APIKeyReadResponseSchema,
    APIKeyCreateSchema,
    APIKeyCreateResponseSchema,
    APIKeyUpdateSchema,
)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


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


class APIKeyService:
    """
    Service layer for managing API keys.

    Attributes:
        _repo (APIKeyRepository): repository for API key operations.
    """

    def __init__(self, repo: APIKeyRepository):
        self._repo = repo

    async def _get_key(self, key_id: int) -> ApiKeyModel:
        """
        Retrieve an API key by its ID.

        Args:
            key_id (int): identifier of the key.

        Returns:
            APIKeyReadResponseSchema: the retrieved key.

        Raises:
            APIKeyNotFoundException: if no such key exists.
        """
        key = await self._repo.get(key_id)

        if not key:
            raise APIKeyNotFoundException(key_id)

        return key

    async def get(self, key_id: int) -> APIKeyReadResponseSchema:
        """
        Retrieve an API key by its ID.

        Args:
            key_id (int): identifier of the key.

        Returns:
            APIKeyReadResponseSchema: the retrieved key.

        Raises:
            APIKeyNotFoundException: if no such key exists.
        """
        key = await self._get_key(key_id)
        return APIKeyReadResponseSchema.model_validate(key)

    async def list_all(self, skip: int = 0, limit: int = 100, active_only: bool = False) -> Sequence[APIKeyReadResponseSchema]:
        """
        List API keys with optional pagination and active‐only filtering.

        Args:
            skip (int): number of records to skip.
            limit (int): maximum number to return.
            active_only (bool): if True, only return active keys.

        Returns:
            List[APIKeyOutputSchema]: list of key schemas.
        """
        keys = await self._repo.list_all(
            skip=skip,
            limit=limit,
            active_only=active_only,
        )
        return [APIKeyReadResponseSchema.model_validate(k) for k in keys]

    async def create(self, data: APIKeyCreateSchema) -> APIKeyCreateResponseSchema:
        """
        Create and persist a new API key.

        Args:
            data (APIKeyCreateSchema): input data.

        Returns:
            APIKeyCreateResponseSchema: the created key + its raw plain_key.
        """
        # Build model (generates & hashes raw_key internally)
        model = ApiKeyModel(name=data.name, description=data.description, is_active=data.is_active)

        # Persist the model
        key = await self._repo.create(model)

        # Retrieve the one-time plain key
        raw_key = key.plain_key

        # Build response schema
        resp = APIKeyCreateResponseSchema.model_validate(key)
        resp.plain_key = raw_key
        return resp

    async def update(self, id_: int, data: APIKeyUpdateSchema) -> APIKeyReadResponseSchema:
        """
        Update fields of an existing API key.

        Args:
            id_ (int): identifier of the key.
            data (APIKeyUpdateSchema): fields to modify.

        Returns:
            APIKeyReadResponseSchema: updated key.

        Raises:
            APIKeyNotFoundException: if no such key exists.
        """
        key = await self._get_key(id_)
        await self._repo.update(key, data.model_dump())
        return APIKeyReadResponseSchema.model_validate(key)

    async def activate(self, key_id: int, active: bool) -> APIKeyReadResponseSchema:
        """
        Activate or deactivate an API key.

        Args:
            key_id (int): identifier of the key.
            active (bool): True to activate, False to deactivate.

        Returns:
            APIKeyReadResponseSchema: the updated key.

        Raises:
            APIKeyNotFoundException: if no such key exists.
        """
        key = await self._get_key(key_id)
        await self._repo.update(key, {"is_active": active})
        return APIKeyReadResponseSchema.model_validate(key)

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

    async def delete(self, key_id: int) -> None:
        """
        Permanently delete an API key by its ID.

        Args:
            key_id (int): identifier of the key to delete.

        Raises:
            APIKeyNotFoundException: if no such key exists.
        """
        key = await self._get_key(key_id)
        await self._repo.delete(key)
