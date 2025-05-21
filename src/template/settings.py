import os
from abc import ABC
from enum import Enum

from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings
from sqlalchemy.engine.url import URL


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

    model_config = ConfigDict(extra="ignore", env_file_encoding="utf-8")

    host: str = Field(default="localhost", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    workers: int = Field(default=1, alias="WORKERS")
    environment: Environment = Field(default=Environment.DEVELOPMENT, alias="ENVIRONMENT")

    postgres_url: URL = Field(..., alias="DATABASE_URL")

    @property
    def database_url(self) -> str:
        """Get the database URL without hiding the password."""
        return self.postgres_url.render_as_string(hide_password=False)


class DevelopmentSettings(Settings):
    postgres_url: URL = Field(
        default=URL.create(
            drivername="sqlite+aiosqlite",
            database="./data/dev.db",
        ),
        alias="DATABASE_URL",
    )


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
