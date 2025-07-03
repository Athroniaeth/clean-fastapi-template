import secrets
from typing import Optional

from passlib.handlers.bcrypt import bcrypt
from pydantic import BaseModel, Field
from pydantic import computed_field
from starlette import status

from template.core.exceptions import APIException

DEFAULT_ROUNDS = 12


def generate_password_hash(raw_password: str) -> str:
    """Generate a password hash using bcrypt."""
    return bcrypt.using(rounds=DEFAULT_ROUNDS).hash(raw_password)


def check_password_hash(hashed_password: str, raw_password: str) -> bool:
    """Check if the provided password matches the hashed password."""
    return bcrypt.verify(raw_password, hashed_password)


def generate_raw_key(length: int = 32) -> str:
    """Generate a random API key."""
    return secrets.token_urlsafe(length)


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


class ApiKey(BaseModel):
    """Domain model for API keys."""

    class Config:
        from_attributes = True

    id_: int = Field(
        default=...,
        alias="id",
    )
    name: str = Field(
        default=...,
        min_length=1,
        max_length=64,
    )
    description: Optional[str] = Field(
        default=None,
        max_length=256,
    )
    __plain_key: str = Field(
        default=...,
        min_length=32,
        max_length=32,
    )
    hashed_key: str = Field(
        default=...,
        min_length=60,
        max_length=60,
    )

    def check_key(self, raw_key: str) -> bool:
        """Check if the provided key matches the hashed key."""
        return check_password_hash(self.hashed_key, raw_key)

    @computed_field
    @property
    def plain_key(self) -> str:
        """Return the plain key, which is only available just after instance creation."""
        # Remove the plain key after use to avoid storing it elsewhere
        plain_key = self.__plain_key
        del self._plain_key
        return plain_key

    """    def __init__(
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
        \"""Check if the provided key matches the hashed key.\"""
        return check_password_hash(self.hashed_key, raw_key)

    @property
    def plain_key(self) -> str:
        \"""
        Give the generated plain key:

        Notes:
         - only available just after instance creation.
         - avoid storing it elsewhere in the database.
        \"""
        # self._plain_key can be no instantiated
        plain_key = self._plain_key

        # Remove the plain key after use to avoid storing it elsewhere
        del self._plain_key
        return plain_key"""
