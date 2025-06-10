from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Mapping

from fastapi import FastAPI as _FastAPI
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request as _Request

from typing import Callable, AsyncContextManager


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
        title (str): The title of the application.
        version (str): The version of the application.
        description (str): The description of the application.
        session (sessionmaker[AsyncSession]): The database session for the application.
    """

    title: str
    version: str
    description: str
    session: sessionmaker

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


type Lifespan = Callable[[], AsyncContextManager[None]]
