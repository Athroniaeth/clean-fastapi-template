from typing import Optional, Sequence

from sqlalchemy import select, update, Column, Integer, String, Boolean, DateTime, func

from template.domain.api_keys import (
    generate_raw_key,
    generate_password_hash,
    check_password_hash,
)
from template.infrastructure.database.base import Base, AbstractDatabaseInfra


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
    """

    def __init__(self, infra_database: AbstractDatabaseInfra) -> None:
        self._infra = infra_database

    async def get(self, id_: int) -> Optional[ApiKeyModel]:
        """
        Retrieve an ApiKeyModel by its integer ID.

        Args:
            id_ (int): primary key of the API key record.

        Returns:
            Optional[ApiKeyModel]: the matching instance, or None if not found.
        """
        async with self._infra.get_session() as session:
            stmt = select(ApiKeyModel).where(ApiKeyModel.id == id_)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Optional[ApiKeyModel]:
        """
        Retrieve an ApiKeyModel by its name field.

        Args:
            name (str): the human-readable name of the API key.

        Returns:
            Optional[ApiKeyModel]: the matching instance, or None if not found.
        """
        async with self._infra.get_session() as session:
            stmt = select(ApiKeyModel).where(ApiKeyModel.name == name)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_by_hashed_key(self, hashed_key: str) -> Optional[ApiKeyModel]:
        """
        Retrieve an ApiKeyModel by its hashed key value.

        Args:
            hashed_key (str): the bcrypt-hashed API key string.

        Returns:
            Optional[ApiKeyModel]: the matching instance, or None if not found.
        """
        async with self._infra.get_session() as session:
            stmt = select(ApiKeyModel).where(ApiKeyModel.hashed_key == hashed_key)
            result = await session.execute(stmt)
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
        async with self._infra.get_session() as session:
            stmt = select(ApiKeyModel)
            if active_only:
                stmt = stmt.where(ApiKeyModel.is_active.is_(True))

            stmt = stmt.offset(skip).limit(limit)
            result = await session.execute(stmt)
            return result.scalars().all()

    async def create(self, api_key: ApiKeyModel) -> ApiKeyModel:
        """
        Insert a new ApiKeyModel into the database.

        Args:
            api_key (ApiKeyModel): the new API key instance to persist.

        Returns:
            ApiKeyModel: the persisted instance, with autogenerated fields populated
        """
        async with self._infra.get_session() as session:
            # Add to the session and flush so that autogenerated IDs are populated
            session.add(api_key)

            await session.commit()
            await session.flush()
            return api_key

    async def activate(self, api_key: ApiKeyModel) -> None:
        """
        Soft-delete an API key by marking it active.

        Args:
            api_key (ApiKeyModel): the instance to activate.
        """
        async with self._infra.get_session() as session:
            api_key.is_active = True
            await session.flush()

    async def deactivate(self, api_key: ApiKeyModel) -> None:
        """
        Soft-delete an API key by marking it inactive.

        Args:
            api_key (ApiKeyModel): the instance to deactivate.
        """
        async with self._infra.get_session() as session:
            api_key.is_active = False
            await session.flush()

    async def update(self, api_key: ApiKeyModel, data: dict) -> bool:
        """
        Update fields of an existing ApiKeyModel.

        Args:
            api_key (ApiKeyModel): the instance to update.
            data (dict): dictionary of fields to modify.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        async with self._infra.get_session() as session:
            stmt = update(ApiKeyModel).where(ApiKeyModel.id == api_key.id).values(**data)
            result = await session.execute(stmt)
            await session.commit()

        # rowcount is bad typed by SQLAlchemy, so we use type: ignore
        return result.rowcount > 0  # type: ignore

    async def delete(self, api_key: ApiKeyModel) -> None:
        """
        Permanently remove an ApiKeyModel from the database.

        Args:
            api_key (ApiKeyModel): the instance to remove.
        """
        async with self._infra.get_session() as session:
            await session.delete(api_key)
            await session.flush()
