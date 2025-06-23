from dataclasses import dataclass, fields
from typing import Mapping

from fastapi import FastAPI as _FastAPI
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request as _Request

from template.core.constants import Lifespan
from template.infrastructure.storage.base import AbstractStorageInfra


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
        async_session (sessionmaker[AsyncSession]): The database session for the application.
    """

    title: str
    version: str
    description: str
    async_session: sessionmaker
    infra_storage: AbstractStorageInfra

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        for f in fields(self):
            yield f.name

    def __len__(self):
        return len(fields(self))


class FastAPI(_FastAPI):
    """Custom FastAPI class to include the HTTP client in the state."""

    title: str
    version: str
    description: str
    lifespan: Lifespan
    state: State


class Request(_Request):
    """Custom request class to include the HTTP client in the state."""

    state: State
