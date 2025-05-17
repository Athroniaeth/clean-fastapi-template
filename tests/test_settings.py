"""
Unit tests for Settings and SQLAlchemySettings classes.

This module contains tests that ensure the Settings class correctly
loads environment variables and provides the appropriate
SQLAlchemySettings subclass based on the ENVIRONMENT configuration.

Example:
    pytest test_settings.py
"""
import os

import pytest
from pydantic import ValidationError

from template.settings import (
    Settings,
    Environment,
    DevelopmentSQLAlchemySettings,
    ProductionSQLAlchemySettings,
)


class TestSettings(Settings):
    """
    Settings class for testing (disables loading from .env).
    """
    class Config:
        env_file = None


def test_default_application_settings(monkeypatch):
    """Verify default application settings when no environment variables are set."""
    # Ensure no overriding environment variables
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("WORKERS", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    settings = TestSettings()

    assert settings.host == "localhost"
    assert settings.port == 8000
    assert settings.workers == 1
    assert settings.environment == Environment.DEVELOPMENT


def test_db_settings_development(monkeypatch):
    """Verify DevelopmentSQLAlchemySettings is returned for development environment."""
    # Set environment to DEVELOPMENT
    monkeypatch.setenv("ENVIRONMENT", Environment.DEVELOPMENT.value)
    
    # Remove any custom DATABASE_URL override
    monkeypatch.delenv("DATABASE_URL", raising=False)

    settings = TestSettings()
    db_settings = settings.db

    assert isinstance(db_settings, DevelopmentSQLAlchemySettings)
    expected = "sqlite+aiosqlite:///./data/dev.db"
    assert db_settings.url == expected


def test_db_settings_production(monkeypatch):
    """Verify ProductionSQLAlchemySettings is returned for production environment."""
    # Set environment to PRODUCTION
    monkeypatch.setenv("ENVIRONMENT", Environment.PRODUCTION.value)
    
    # Remove any custom DATABASE_URL override
    monkeypatch.delenv("DATABASE_URL", raising=False)

    settings = TestSettings()
    db_settings = settings.db

    assert isinstance(db_settings, ProductionSQLAlchemySettings)
    expected = "postgresql+asyncpg://username:password@postgres:5432/database"
    assert db_settings.url == expected


def test_invalid_environment_raises(monkeypatch):
    """Verify that an unsupported environment setting raises a KeyError."""
    # Set an invalid environment value
    monkeypatch.setenv("ENVIRONMENT", "unsupported_env")
    
    with pytest.raises(ValidationError):
        _ = TestSettings()
        