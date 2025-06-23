from typing import Optional, Sequence

from sqlalchemy import select, update, Column, Integer, String, Boolean, DateTime, func
from sqlalchemy.ext.asyncio import AsyncSession

from template.domain.api_keys import (
    generate_raw_key,
    generate_password_hash,
    check_password_hash,
    APIKeyNotFoundException,
    APIKeyNotProvidedException,
    APIKeyInvalidException,
)
from template.infrastructure.database.base import Base
from template.schemas.api_keys import (
    APIKeyReadResponseSchema,
    APIKeyCreateSchema,
    APIKeyCreateResponseSchema,
    APIKeyUpdateSchema,
)


class ApiKeyModel(Base):
    __tablename__ = "api_keys"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    name = Column(
        String(64),
        nullable=False,
    )
    description = Column(
        String(255),
        nullable=True,
    )
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    hashed_key = Column(
        "hashed_key",
        String(128),
        nullable=False,
        unique=True,
    )

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        is_active: bool = True,
    ):
        super().__init__()
        self.name = name
        self.description = description
        self.is_active = is_active

        # Generate a random API key and hash it
        raw_key = generate_raw_key()
        self.hashed_key = generate_password_hash(raw_key)

        # Create a temporary attribute to store the raw key
        self._plain_key = raw_key

    def check_key(self, raw_key: str) -> bool:
        """Check if the provided key matches the hashed key."""
        return check_password_hash(self.hashed_key, raw_key)

    @property
    def plain_key(self) -> str:
        """
        Give the generated plain key:

        Notes:
         - only available just after instance creation.
         - avoid storing it elsewhere in the database.
        """
        # self._plain_key can be no instantiated
        plain_key = self._plain_key

        # Remove the plain key after use to avoid storing it elsewhere
        del self._plain_key
        return plain_key


class APIKeyRepository:
    """
    Concrete repository for managing ApiKeyModel persistence with SQLAlchemy.

    Attributes:
        _session (AsyncSession): SQLAlchemy session for database operations.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get(self, id_: int) -> Optional[ApiKeyModel]:
        """
        Retrieve an ApiKeyModel by its integer ID.

        Args:
            id_ (int): primary key of the API key record.

        Returns:
            Optional[ApiKeyModel]: the matching instance, or None if not found.
        """
        stmt = select(ApiKeyModel).where(ApiKeyModel.id == id_)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Optional[ApiKeyModel]:
        """
        Retrieve an ApiKeyModel by its name field.

        Args:
            name (str): the human-readable name of the API key.

        Returns:
            Optional[ApiKeyModel]: the matching instance, or None if not found.
        """
        stmt = select(ApiKeyModel).where(ApiKeyModel.name == name)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_hashed_key(self, hashed_key: str) -> Optional[ApiKeyModel]:
        """
        Retrieve an ApiKeyModel by its hashed key value.

        Args:
            hashed_key (str): the bcrypt-hashed API key string.

        Returns:
            Optional[ApiKeyModel]: the matching instance, or None if not found.
        """
        stmt = select(ApiKeyModel).where(ApiKeyModel.hashed_key == hashed_key)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(self, skip: int = 0, limit: int = 100, active_only: bool = False) -> Sequence[ApiKeyModel]:
        """
        List ApiKeyModel records, with optional pagination and active filtering.

        Args:
            skip (int): number of records to skip (for pagination).
            limit (int): maximum number of records to return.
            active_only (bool): if True, only return records where is_active == True.

        Returns:
            Sequence[ApiKeyModel]: list of ApiKeyModel instances.
        """
        stmt = select(ApiKeyModel)
        if active_only:
            stmt = stmt.where(ApiKeyModel.is_active.is_(True))

        stmt = stmt.offset(skip).limit(limit)
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def create(self, api_key: ApiKeyModel) -> ApiKeyModel:
        """
        Insert a new ApiKeyModel into the database.

        Args:
            api_key (ApiKeyModel): the new API key instance to persist.

        Returns:
            ApiKeyModel: the persisted instance, with autogenerated fields populated
        """
        # Add to the session and flush so that autogenerated IDs are populated
        self._session.add(api_key)

        await self._session.commit()
        await self._session.flush()
        return api_key

    async def activate(self, api_key: ApiKeyModel) -> None:
        """
        Soft-delete an API key by marking it active.

        Args:
            api_key (ApiKeyModel): the instance to activate.
        """
        api_key.is_active = True
        await self._session.flush()

    async def deactivate(self, api_key: ApiKeyModel) -> None:
        """
        Soft-delete an API key by marking it inactive.

        Args:
            api_key (ApiKeyModel): the instance to deactivate.
        """
        api_key.is_active = False
        await self._session.flush()

    async def update(self, api_key: ApiKeyModel, data: dict) -> bool:
        """
        Update fields of an existing ApiKeyModel.

        Args:
            api_key (ApiKeyModel): the instance to update.
            data (dict): dictionary of fields to modify.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        stmt = update(ApiKeyModel).where(ApiKeyModel.id == api_key.id).values(**data)
        result = await self._session.execute(stmt)
        await self._session.commit()

        # rowcount is bad typed by SQLAlchemy, so we use type: ignore
        return result.rowcount > 0  # type: ignore

    async def delete(self, api_key: ApiKeyModel) -> None:
        """
        Permanently remove an ApiKeyModel from the database.

        Args:
            api_key (ApiKeyModel): the instance to remove.
        """
        await self._session.delete(api_key)
        await self._session.flush()


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
