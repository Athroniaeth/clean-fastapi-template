from enum import Enum

from pydantic import Field, computed_field, PrivateAttr
from pydantic_settings import BaseSettings
from sqlalchemy.engine.url import URL


class Environment(str, Enum):
    """
    Environment to use.
    - DEVELOPMENT: Development with SQLite
    - PRODUCTION: Docker compose with PostgreSQL
    """

    STAGING = "staging"
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """
    Base settings for the application.
    """

    host: str = Field(default="localhost", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    workers: int = Field(default=1, alias="WORKERS")
    environment: Environment = Field(default=Environment.DEVELOPMENT, alias="ENVIRONMENT")

    class Config:
        env_file_encoding = "utf-8"

    @property
    def db(self) -> "SQLAlchemySettings":
        """
        Get the database settings based on the environment.
        """
        mapping: dict[Environment, type[SQLAlchemySettings]] = {
            Environment.DEVELOPMENT: DevelopmentSQLAlchemySettings,
            Environment.PRODUCTION: ProductionSQLAlchemySettings,
        }
        cls = mapping[self.environment]
        return cls()


class SQLAlchemySettings(BaseSettings):
    sqlalchemy_url: URL = Field(..., alias="DATABASE_URL")
    
    @computed_field
    @property
    def url(self) -> str:
        """Get the database URL without hiding the password."""
        return self.sqlalchemy_url.render_as_string(hide_password=False)

    class Config:
        env_file_encoding = "utf-8"


class DevelopmentSQLAlchemySettings(SQLAlchemySettings):
    sqlalchemy_url: URL = Field(
        default=URL.create(
            drivername="sqlite+aiosqlite",
            database="./data/dev.db",
        ),
        alias="DATABASE_URL",
    )


class ProductionSQLAlchemySettings(SQLAlchemySettings):
    sqlalchemy_url: URL = Field(
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
