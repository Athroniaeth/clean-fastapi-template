from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError

from template.core.exceptions import APIException
from template.infrastructure.database.adapter import SQLiteDatabaseInfra
from template.infrastructure.database.users import UserRepository, UserService
from template.schemas.users import UserReadResponse, UserCreateSchema
from template.domain.users import UserNotFoundException


class JWTCredentialException(APIException):
    """Base class for JWT credential-related exceptions."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)
        self.status_code = status_code
        self.detail = detail


class JWTCredentialInvalidException(JWTCredentialException):
    """Raised when JWT credentials are invalid or missing."""

    status_code = status.HTTP_401_UNAUTHORIZED
    detail = "Invalid or missing JWT credentials."

    def __init__(self):
        super().__init__(
            status_code=self.status_code,
            detail=self.detail,
        )


# JWT Security scheme for local password authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


class AuthService:
    """
    Service responsible for JWT-based authentication.

    Attributes:
        _user_service (UserService): service for accessing user data.
    """

    def __init__(self, user_service: UserService, jwt_secret_key: str, jwt_algorithm: str, jwt_exp_minutes: int = 15):
        self._user_service = user_service
        self._jwt_secret_key = jwt_secret_key
        self._jwt_algorithm = jwt_algorithm
        self._jwt_exp_minutes = jwt_exp_minutes

    def _create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Generate a JWT token from provided payload."""
        to_encode = data.copy()
        expire = datetime.now() + (expires_delta or timedelta(minutes=self._jwt_exp_minutes))
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self._jwt_secret_key, algorithm=self._jwt_algorithm)
        return encoded_jwt

    async def login(self, username: str, password: str) -> str:
        """Authenticate user and return JWT token.

        Args:
            username (str): Username.
            password (str): Plain password.

        Returns:
            str: JWT token.

        Raises:
            HTTPException: If credentials are invalid.
        """
        await self._user_service.verify_password(username, password)
        return self._create_access_token({"sub": username})

    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> UserReadResponse:
        """Dependency to retrieve current user from token.

        Args:
            token (str): JWT token from Authorization header.

        Returns:
            UserReadResponse: Authenticated user.

        Raises:
            HTTPException: If token is invalid or user not found.
        """
        try:
            payload = jwt.decode(token, self._jwt_secret_key, algorithms=[self._jwt_algorithm])
            username: Optional[str] = payload.get("sub")
            if not username:
                raise JWTCredentialInvalidException()
        except JWTError:
            raise JWTCredentialInvalidException()

        try:
            user_model = await self._user_service.get_by_username(username)
        except UserNotFoundException:
            raise JWTCredentialInvalidException()

        return UserReadResponse.model_validate(user_model)

    async def register(self, schema: UserCreateSchema) -> UserReadResponse:
        await self._user_service.create(schema)
        token = await self.login(username=schema.username, password=schema.raw_password)
        user = await self.get_current_user(token=token)
        return user


def get_user_service() -> UserService:
    """
    Provide a new UserService instance.
    """
    infra = SQLiteDatabaseInfra()
    repo = UserRepository(infra)
    return UserService(repo)


def get_auth_service(user_service: UserService = Depends(get_user_service)) -> AuthService:
    """
    Provide a new AuthService instance with its dependencies resolved.
    """
    return AuthService(
        user_service,
        jwt_secret_key="your_jwt_secret_key",
        jwt_algorithm="HS256",
        jwt_exp_minutes=15,
    )


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserReadResponse:
    """
    Decode the token and retrieve the corresponding user.
    """
    return await auth_service.get_current_user(token)
