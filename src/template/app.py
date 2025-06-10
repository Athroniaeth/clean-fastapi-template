from contextlib import asynccontextmanager
from functools import partial
from typing import AsyncIterator

from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from loguru import logger

from template.core.state import FastAPI, get_version, State
from template.core.exceptions import APIException
from template.infrastructure.database import create_db
from template.routes.index import index_router, api_router
from template.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI, database_url: str) -> AsyncIterator[State]:
    """Basic lifespan context manager for FastAPI."""

    # Note: We log with TRACE for not spam with pytest
    logger.debug("Starting FastAPI application lifecycle")

    # Create an async engine
    async_session = await create_db(database_url)

    state: State = State(
        title=app.title,
        version=app.version,
        description=app.description,
        session=async_session,
    )
    yield state

    logger.debug("Shutting down FastAPI application lifecycle")


def create_app(
    title: str,
    version: str,
    description: str,
    lifespan: asynccontextmanager,
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
        lifespan=partial(lifespan, database_url=settings.database_url),
    )
    return app
