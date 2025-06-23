from passlib.handlers.bcrypt import bcrypt
from starlette import status

from template.core.exceptions import APIException


class UserException(APIException):
    """Base class for User related exceptions."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)
        self.status_code = status_code
        self.detail = detail


class UserAlreadyExistsException(UserException):
    """Raised when trying to create a User that already exists."""

    status_code = status.HTTP_409_CONFLICT
    detail = "User with username '{username}' already exists."

    def __init__(self, username: str):
        super().__init__(
            status_code=self.status_code,
            detail=self.detail.format(username=username),
        )


class UserNotFoundException(UserException):
    """Raised when a User is not found in the database."""

    status_code = status.HTTP_404_NOT_FOUND
    detail = "User not found : {user_id_or_name}"

    def __init__(self, user_id_or_name: int | str):
        super().__init__(
            status_code=self.status_code,
            detail=self.detail.format(user_id_or_name=user_id_or_name),
        )


class UserNotProvidedException(UserException):
    """Raised when a User is not provided in the request."""

    status_code = status.HTTP_401_UNAUTHORIZED
    detail = "User not provided in the request."

    def __init__(self):
        super().__init__(
            status_code=self.status_code,
            detail=self.detail,
        )


class PasswordNotProvidedException(UserException):
    """Raised when a raw password is not provided in the request."""

    status_code = status.HTTP_401_UNAUTHORIZED
    detail = "Raw password not provided in the request."

    def __init__(self):
        super().__init__(
            status_code=self.status_code,
            detail=self.detail,
        )


class PasswordInvalidException(UserException):
    """Raised when a raw password is invalid or does not match any stored hashes."""

    status_code = status.HTTP_401_UNAUTHORIZED
    detail = "Invalid password provided : {raw_password}"

    def __init__(self, raw_password: str):
        super().__init__(
            status_code=self.status_code,
            detail=self.detail.format(raw_password=raw_password),
        )


DEFAULT_ROUNDS = 12


def generate_password_hash(raw_password: str) -> str:
    """Generate a password hash using bcrypt."""
    return bcrypt.using(rounds=DEFAULT_ROUNDS).hash(raw_password)


def check_password_hash(hashed_password: str, raw_password: str) -> bool:
    """Check if the provided password matches the hashed password."""
    return bcrypt.verify(raw_password, hashed_password)
