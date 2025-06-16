from contextlib import asynccontextmanager
from functools import partial
from typing import AsyncIterator

from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from loguru import logger

from template.core.constants import State, FastAPI, Lifespan
from template import get_version
from template.core.exceptions import APIException
from template.infrastructure.sql.base import create_db
from template.routes.index import index_router, api_router
from template.settings import get_settings, Settings, get_storage_infra


@asynccontextmanager
async def lifespan(app: FastAPI, settings: Settings) -> AsyncIterator[State]:
    """Basic lifespan context manager for FastAPI."""

    # Note: We log with TRACE for not spam with pytest
    logger.debug("Starting FastAPI application lifecycle")

    # Create an async engine
    async_session = await create_db(settings.database_url)

    # Initialize the storage (file) infrastructure
    infra_storage = get_storage_infra(settings)
    # Todo: Hard to test, need found solution
    # await s3_client.ensure_bucket_exists()

    state: State = State(
        title=app.title,
        version=app.version,
        description=app.description,
        async_session=async_session,
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
    # Load settings
    settings = get_settings()

    # Create the FastAPI app
    app = create_app(
        title="Template FastAPI App",
        version=get_version(),
        description="A template FastAPI application with async capabilities.",
        lifespan=partial(lifespan, settings=settings),
    )
    return app
