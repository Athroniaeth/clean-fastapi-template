import os
from abc import ABC
from enum import Enum
from pathlib import Path

from loguru import logger
from pydantic import Field, ConfigDict, HttpUrl
from pydantic_settings import BaseSettings
from sqlalchemy.engine.url import URL

from rename import DATA_PATH
from rename.infrastructure.database.base import AbstractDatabaseInfra, Base
from rename.infrastructure.storage.base import AbstractStorageInfra


class SettingsError(Exception):
    """Custom exception for settings errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidEnvironmentError(SettingsError):
    """Custom exception for invalid environment errors."""

    def __init__(self, environment: str):
        available_envs = [env.value for env in Environment]
        available_envs = ", ".join(available_envs)
        message = f"Invalid environment: '{environment}', expected one of {available_envs}"

        super().__init__(message)
        self.environment = environment


class MissingEnvironmentError(SettingsError):
    """Custom exception for missing environment errors."""

    def __init__(self, variable: str):
        super().__init__(f"Missing environment variable: '{variable}'")
        self.variable = variable


class Environment(str, Enum):
    """
    Environment to use.
    - DEVELOPMENT: Development with SQLite
    - PRODUCTION: Docker compose with PostgreSQL
    """

    STAGING = "staging"
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class Settings(ABC, BaseSettings):
    """
    Base settings for the application.
    """

    model_config = ConfigDict(
        extra="ignore",
        env_file_encoding="utf-8",
    )

    host: str = Field(default="localhost", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    workers: int = Field(default=1, alias="WORKERS")
    environment: Environment = Field(default=Environment.DEVELOPMENT, alias="ENVIRONMENT")

    postgres_url: URL = Field(..., alias="DATABASE_URL")

    local_storage: bool = Field(default=True, alias="LOCAL_STORAGE")

    s3_bucket: str = Field(..., alias="S3_BUCKET")
    s3_region: str = Field(default="us-east-1", alias="S3_REGION")
    s3_endpoint_url: HttpUrl | None = Field(default=None, alias="S3_ENDPOINT_URL")  # Optional for localstack/minio
    s3_access_key_id: str = Field(..., alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str = Field(..., alias="S3_SECRET_ACCESS_KEY")

    @property
    def database_url(self) -> str:
        """Get the database URL without hiding the password."""
        return self.postgres_url.render_as_string(hide_password=False)


class DevelopmentSettings(Settings):
    postgres_url: URL = Field(
        default=URL.create(
            drivername="sqlite+aiosqlite",
            database="./data/db.sqlite",
        ),
        alias="DATABASE_URL",
    )

    local_storage: bool = Field(default=True, alias="LOCAL_STORAGE")

    s3_bucket: str = Field(default="test-bucket", alias="S3_BUCKET")
    s3_region: str = Field(default="eu-west-1", alias="S3_REGION")
    s3_endpoint_url: HttpUrl = Field(default="http://localhost:5000", alias="S3_ENDPOINT_URL")  # ty: ignore[invalid-assignment]
    s3_access_key_id: str = Field(default="None", alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str = Field(default="None", alias="S3_SECRET_ACCESS_KEY")


class ProductionSettings(Settings):
    postgres_url: URL = Field(
        default=URL.create(
            drivername="postgresql+asyncpg",
            username="username",
            password="password",
            host="postgres",
            port=5432,
            database="database",
        ),
        alias="DATABASE_URL",
    )

    local_storage: bool = Field(default=False, alias="LOCAL_STORAGE")

    s3_bucket: str = Field(default="test-bucket", alias="S3_BUCKET")
    s3_region: str = Field(default="eu-west-1", alias="S3_REGION")
    s3_endpoint_url: HttpUrl = Field(default="http://localhost:5000", alias="S3_ENDPOINT_URL")  # ty: ignore[invalid-assignment]
    s3_access_key_id: str = Field(default="None", alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str = Field(default="None", alias="S3_SECRET_ACCESS_KEY")


def get_settings() -> Settings:
    """Get the settings class based on the environment."""
    mapping = {
        Environment.DEVELOPMENT: DevelopmentSettings,
        Environment.PRODUCTION: ProductionSettings,
    }

    # Get the environment from the environment variable
    environment = os.getenv("ENVIRONMENT")

    if environment is None:
        raise MissingEnvironmentError("ENVIRONMENT")

    settings_class = mapping.get(environment)

    if settings_class is None:
        raise InvalidEnvironmentError(environment)

    # Return the settings class instance
    return settings_class()


def get_database_infra(settings: Settings) -> AbstractDatabaseInfra:
    """Get the database settings based on the provided settings."""

    from rename.infrastructure.database.adapter import PostgresDatabaseInfra, SQLiteDatabaseInfra

    if settings.environment == Environment.DEVELOPMENT:
        return SQLiteDatabaseInfra(base=Base)

    return PostgresDatabaseInfra(base=Base)


def get_storage_infra(settings: Settings) -> AbstractStorageInfra:
    """
    Get the storage settings based on the provided settings.
    This is useful for testing purposes.
    """
    from rename.infrastructure.storage.s3 import S3StorageInfra

    from rename.infrastructure.storage.local import LocalStorageInfra

    if settings.local_storage:
        base_path = DATA_PATH.relative_to(Path(os.getcwd()).parent)
        logger.debug("Using LocalStorageInfra infrastructure")
        logger.debug(f"LocalStorageInfra using base path '{base_path}'")
        return LocalStorageInfra(base_path=DATA_PATH)

    logger.debug(f"Using S3 storage infrastructure with bucket: '{settings.s3_bucket}'")
    return S3StorageInfra(
        bucket_name=settings.s3_bucket,
        region_name=settings.s3_region,
        endpoint_url=f"{settings.s3_endpoint_url}",
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
    )
