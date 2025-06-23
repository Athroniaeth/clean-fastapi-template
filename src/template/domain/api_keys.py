import secrets

from passlib.handlers.bcrypt import bcrypt
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
