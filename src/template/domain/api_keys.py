import secrets
from typing import Optional

from passlib.handlers.bcrypt import bcrypt
from pydantic import BaseModel, Field, PrivateAttr
from pydantic import computed_field
from starlette import status

from template.core.exceptions import APIException

DEFAULT_ROUNDS = 4


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

    id_: int = Field(
        default=None,
    )
    name: str = Field(
        default=...,
        min_length=1,
        max_length=64,
    )
    description: Optional[str] = Field(
        default="N/A",
        max_length=256,
    )
    is_active: bool = Field(
        default=True,
    )
    # allow None here so BaseModel __init__ doesn’t complain
    hashed_key: str = Field(
        default=None,
        min_length=60,
        max_length=60,
    )

    _plain_key: Optional[str] = PrivateAttr(
        default=None,
    )

    class Config:
        from_attributes = True

    @staticmethod
    def generate_raw_key() -> str:
        """Generate a new raw API key."""
        return generate_raw_key()

    @staticmethod
    def generate_hashed_key(raw_key: str) -> str:
        """Generate a hashed API key from the raw key."""
        return generate_password_hash(raw_key)

    @classmethod
    def create(
        cls,
        name: str,
        description: Optional[str] = None,
        is_active: bool = True,
    ) -> "ApiKey":
        """Factory method to create an ApiKey with generated key and hash."""
        raw_key = ApiKey.generate_raw_key()
        hashed_key = ApiKey.generate_hashed_key(raw_key)

        instance = cls(
            name=name,
            description=description,
            is_active=is_active,
            hashed_key=hashed_key,
        )
        instance._plain_key = raw_key
        return instance

    """    def __init__(self, name: str, description: Optional[str] = None, is_active: bool = True, id_: Optional[int] = None):
        ""Initialize the API key with a name, description, and active status.""
        # Generate a random API key and hash it
        raw_key = generate_raw_key()
        hashed_key = generate_password_hash(raw_key)

        super().__init__(
            id_=id_,
            name=name,
            description=description,
            is_active=is_active,
            hashed_key=hashed_key,
        )
        self._plain_key = raw_key"""

    def __repr__(self) -> str:
        """Return a string representation of the API key."""
        return f"ApiKey(id_={self.id_}, name={self.name}, is_active={self.is_active})"

    def check_key(self, raw_key: str) -> bool:
        """Check if the provided key matches the hashed key."""
        return check_password_hash(self.hashed_key, raw_key)

    @computed_field
    @property
    def plain_key(self) -> str:
        """Return the plain key, which is only available just after instance creation."""
        # Remove the plain key after use to avoid storing it elsewhere
        plain_key = self._plain_key

        if not plain_key:
            raise ValueError("Plain key is not available.")

        del self._plain_key
        return plain_key
