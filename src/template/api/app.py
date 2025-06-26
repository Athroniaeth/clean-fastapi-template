from loguru import logger

from contextlib import asynccontextmanager
from functools import partial
from typing import AsyncIterator
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from template.core.constants import Lifespan
from template.core.improve import State, FastAPI
from template import get_version
from template.core.exceptions import APIException

from template.api.routes.v1.router import index_router, api_router

from template.infrastructure.storage.base import AbstractStorageInfra
from template.infrastructure.database.base import AbstractDatabaseInfra

from template.settings import get_settings, Settings, get_storage_infra, get_database_infra


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    settings: Settings,
    infra_storage: AbstractStorageInfra,
    infra_database: AbstractDatabaseInfra,
) -> AsyncIterator[State]:
    """Basic lifespan context manager for FastAPI."""

    # Note: We log with TRACE for not spam with pytest
    logger.debug("Starting FastAPI application lifecycle")

    state = State(
        title=app.title,
        version=app.version,
        description=app.description,
        settings=settings,
        infra_database=infra_database,
        infra_storage=infra_storage,
    )
    yield state

    logger.debug("Shutting down FastAPI application lifecycle")


def create_app(
    title: str,
    version: str,
    description: str,
    lifespan: Lifespan,
) -> FastAPI:
    """Create a new instance of the application."""
    app = FastAPI(
        title=title,
        version=version,
        lifespan=lifespan,
        description=description,
        default_response_class=ORJSONResponse,  # Improve performance with ORJSONResponse
    )

    # Improve performance with GZip compression
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1000,
    )

    # Disable CORS for all origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add routes for the API
    app.include_router(router=index_router)
    app.include_router(router=api_router)

    app.add_exception_handler(
        APIException,
        http_exception_handler,
    )

    return app


def factory_app() -> FastAPI:
    """Create the FastAPI application."""
    # Load settings from config file
    settings = get_settings()

    # Initialize the infrastructure
    infra_storage = get_storage_infra(settings)
    infra_database = get_database_infra(settings)

    # Todo: Hard to test, need found solution
    # await s3_client.ensure_bucket_exists()

    # Create the FastAPI app
    app = create_app(
        title="Template FastAPI App",
        version=get_version(),
        description="A template FastAPI application with async capabilities.",
        lifespan=partial(
            lifespan,
            settings=settings,
            infra_database=infra_database,
            infra_storage=infra_storage,
        ),
    )
    return app
