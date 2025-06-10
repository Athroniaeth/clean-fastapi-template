"""
Unit tests for Settings and SQLAlchemySettings classes.

This module contains tests that ensure the Settings class correctly
loads environment variables and provides the appropriate
SQLAlchemySettings subclass based on the ENVIRONMENT configuration.

Example:
    pytest test_settings.py
"""

import pytest
from pydantic import Field, ConfigDict, HttpUrl
from sqlalchemy import URL

from template.settings import (
    Settings,
    Environment,
    DevelopmentSettings,
    ProductionSettings,
    get_settings,
    InvalidEnvironmentError,
    MissingEnvironmentError,
)


class CustomSettings(Settings):
    """
    Settings class for testing (disables loading from .env).

    Notes:
        Name this class "TestSettings" triggered warning with pytest-xdist.
    """

    postgres_url: URL = Field(
        default=URL.create(
            drivername="sqlite+aiosqlite",
            database=":memory:",
        ),
        alias="DATABASE_URL",
    )

    s3_bucket: str = Field(default="test-bucket", alias="S3_BUCKET")
    s3_region: str = Field(default="eu-west-1", alias="S3_REGION")
    s3_endpoint_url: HttpUrl = Field(default="http://localhost:5000", alias="S3_ENDPOINT_URL")  # ty: ignore[invalid-assignment]
    s3_access_key_id: str = Field(default="None", alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str = Field(default="None", alias="S3_SECRET_ACCESS_KEY")

    model_config = ConfigDict(env_file=None)


def test_default_application_settings(monkeypatch):
    """Verify default application settings when no environment variables are set."""
    # Ensure no overriding environment variables
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("WORKERS", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    settings = CustomSettings()

    assert settings.host == "localhost"
    assert settings.port == 8000
    assert settings.workers == 1
    assert settings.environment == Environment.DEVELOPMENT


def test_db_settings_development(monkeypatch):
    """Verify DevelopmentSettings is returned for development environment."""
    # Set environment to DEVELOPMENT
    monkeypatch.setenv("ENVIRONMENT", Environment.DEVELOPMENT.value)

    # Remove any custom DATABASE_URL override
    monkeypatch.delenv("DATABASE_URL", raising=False)

    # Get settings after loading environment variables
    settings = get_settings()

    assert isinstance(settings, DevelopmentSettings)
    expected = "sqlite+aiosqlite:///./data/dev.db"
    assert settings.database_url == expected


def test_db_settings_production(monkeypatch):
    """Verify ProductionSettings is returned for production environment."""
    # Set environment to PRODUCTION
    monkeypatch.setenv("ENVIRONMENT", Environment.PRODUCTION.value)

    # Remove any custom DATABASE_URL override
    monkeypatch.delenv("DATABASE_URL", raising=False)

    # Get settings after loading environment variables
    settings = get_settings()

    assert isinstance(settings, ProductionSettings)
    expected = "postgresql+asyncpg://username:password@postgres:5432/database"
    assert settings.database_url == expected


def test_missing_environment_raises(monkeypatch):
    """Verify that missing ENVIRONMENT variable raises a ValueError."""
    # Remove ENVIRONMENT variable
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    with pytest.raises(MissingEnvironmentError):
        _ = get_settings()


def test_invalid_environment_raises(monkeypatch):
    """Verify that an unsupported environment setting raises a KeyError."""
    # Set an invalid environment value
    monkeypatch.setenv("ENVIRONMENT", "unsupported_env")

    with pytest.raises(InvalidEnvironmentError):
        _ = get_settings()
