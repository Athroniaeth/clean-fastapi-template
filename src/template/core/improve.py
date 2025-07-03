from dataclasses import dataclass, fields
from typing import Mapping

from fastapi import FastAPI as _FastAPI
from starlette.requests import Request as _Request


from template.infrastructure.database.base import AbstractDatabaseInfra
from template.infrastructure.storage.base import AbstractStorageInfra
from template.interface.cli.app import Lifespan
from template.settings import Settings


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

        settings (Settings): The settings of the application.
        infra_storage (AbstractStorageInfra): The storage infrastructure.
        infra_database (AbstractDatabaseInfra): The database infrastructure.
    """

    title: str
    version: str
    description: str

    settings: Settings
    infra_storage: AbstractStorageInfra
    infra_database: AbstractDatabaseInfra

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

    state: State
    lifespan: Lifespan


class Request(_Request):
    """Custom request class to include the HTTP client in the state."""

    state: State
