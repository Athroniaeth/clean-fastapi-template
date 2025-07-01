from typing import Sequence

from template.interface.api.schemas.users import (
    UserReadResponseSchema,
    UserCreateSchema,
    UserReadResponse,
    UserUpdateSchema,
)
from template.domain.users import (
    UserNotFoundException,
    UserAlreadyExistsException,
    UserNotProvidedException,
    PasswordNotProvidedException,
    PasswordInvalidException,
)
from template.infrastructure.repositories.users import UserRepository, UserModel


class UserService:
    """
    Service layer for managing API keys.

    Attributes:
        _repo (UserRepository): repository for User operations.
    """

    def __init__(self, repo: UserRepository):
        self._repo = repo

    async def _get(self, user_id: int) -> UserModel:
        """
        Retrieve an User by its ID.

        Args:
            user_id (int): identifier of the user.

        Returns:
            UserReadResponseSchema: the retrieved user.

        Raises:
            UserNotFoundException: if no such user exists.
        """
        user = await self._repo.get(user_id)

        if not user:
            raise UserNotFoundException(user_id)

        return user

    async def get_by_username(self, username: str) -> UserModel:
        """
        Retrieve an User by its username.

        Args:
            username (str): the human-readable name of the user.

        Returns:
            Optional[UserModel]: the matching instance, or None if not found.
        """
        user = await self._repo.get_by_username(username)

        if not user:
            raise UserNotFoundException(username)

        return user

    async def get(self, user_id: int) -> UserReadResponseSchema:
        """
        Retrieve an User by its ID.

        Args:
            user_id (int): identifier of the user.

        Returns:
            UserReadResponseSchema: the retrieved user.

        Raises:
            UserNotFoundException: if no such user exists.
        """
        user = await self._get(user_id)
        return UserReadResponseSchema.model_validate(user)

    async def list_all(self, skip: int = 0, limit: int = 100) -> Sequence[UserReadResponseSchema]:
        """
        List Users with optional pagination and active‐only filtering.

        Args:
            skip (int): number of records to skip.
            limit (int): maximum number to return.

        Returns:
            List[UserOutputSchema]: list of user schemas.
        """
        users = await self._repo.list_all(
            skip=skip,
            limit=limit,
        )
        return [UserReadResponseSchema.model_validate(k) for k in users]

    async def create(self, data: UserCreateSchema) -> UserReadResponse:
        """
        Create and persist a new User.

        Args:
            data (UserCreateSchema): input data.

        Returns:
            UserCreateResponseSchema: the created user + its raw plain_user.
        """
        # Check if user already exists (handle sql integrity errors)
        if await self._repo.get_by_username(data.username):
            raise UserAlreadyExistsException(data.username)

        # Build model (generates & hashes raw_user internally)
        model = UserModel(username=data.username, raw_password=data.raw_password)

        # Persist the model
        user = await self._repo.create(model)

        # Build response schema
        return UserReadResponse.model_validate(user)

    async def update(self, id_: int, data: UserUpdateSchema) -> UserReadResponseSchema:
        """
        Update fields of an existing User.

        Args:
            id_ (int): identifier of the user.
            data (UserUpdateSchema): fields to modify.

        Returns:
            UserReadResponseSchema: updated user.

        Raises:
            UserNotFoundException: if no such user exists.
        """
        user = await self._get(id_)
        await self._repo.update(user, data.model_dump())
        return UserReadResponseSchema.model_validate(user)

    async def verify_password(self, username: str, raw_password: str):
        """
        Verify that a raw User is valid against stored hashes.

        Only active users are checked.

        Args:
            username (str): the human-readable name of the user.
            raw_password (str): the plain user to verify.

        Returns:
            bool: True if valid.

        Raises:
            UserInvalidException: if no match is found.
        """
        if not username:
            raise UserNotProvidedException()

        if not raw_password:
            raise PasswordNotProvidedException()

        user = await self.get_by_username(username)

        if not user:
            raise UserNotFoundException(username)

        if not user.check_password(raw_password):
            raise PasswordInvalidException(raw_password)

    async def delete(self, user_id: int) -> None:
        """
        Permanently delete an User by its ID.

        Args:
            user_id (int): identifier of the user to delete.

        Raises:
            UserNotFoundException: if no such user exists.
        """
        user = await self._get(user_id)
        await self._repo.delete(user)
