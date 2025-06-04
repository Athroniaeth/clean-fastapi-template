import tomllib
from contextlib import asynccontextmanager
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Mapping, AsyncGenerator, Any  # type: ignore[import]

from fastapi import FastAPI as _FastAPI
from fastapi import Request as _Request
from httpx import AsyncClient
from loguru import logger

from template import PROJECT_PATH
from template.infrastructure.database import create_database


class Level(StrEnum):
    """
    Log levels used to trace application execution.

    Attributes:
        TRACE   : Very fine-grained details for deep debugging.
        DEBUG   : Debugging information for developers.
        INFO    : General operational events.
        WARNING : Unexpected behavior that doesn't stop execution.
        ERROR   : Critical issues that affect functionality.
    """

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class State(Mapping):
    """
    State for the FastAPI application.

    Notes:
        Have metaclass TypedDict with dataclass allow to
        have typing with attributes and not only with keys.
        See: https://github.com/Kludex/fastapi-tips?tab=readme-ov-file#6-use-lifespan-state-instead-of-appstate

        We integrate `Mapping` to allow this class to be used by FastAPI

    Attributes:
        client (AsyncClient): The HTTP client used for making requests.
        title (str): The title of the application.
        version (str): The version of the application.
        description (str): The description of the application.
    """

    title: str
    version: str
    description: str
    client: AsyncClient

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        for f in fields(self):
            yield f.name

    def __len__(self):
        return len(fields(self))


class FastAPI(_FastAPI):
    """Custom FastAPI class to include the HTTP client in the state."""

    state: State


class Request(_Request):
    """Custom request class to include the HTTP client in the state."""

    state: State


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[State, Any]:
    """Basic lifespan context manager for FastAPI."""

    # Create database connection
    await create_database()

    # Note: We log with TRACE for not spam with pytest
    logger.debug("Starting FastAPI application lifecycle")

    async with AsyncClient() as client:
        state: State = State(
            client=client,
            title=app.title,
            version=app.version,
            description=app.description,
        )
        yield state

    logger.debug("Shutting down FastAPI application lifecycle")


def get_version() -> str:
    """Get the version of the application."""
    # Get the version of the pyproject.toml file
    pyproject_path = PROJECT_PATH / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")
    dict_content = tomllib.loads(content)
    return dict_content["project"]["version"]
