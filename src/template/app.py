from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

from dotenv import load_dotenv
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from httpx import AsyncClient
from loguru import logger

from template import get_version
from template.constants import State, FastAPI
from template.core.exceptions import APIException
from template.routers.index import index_router, api_router


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[State, Any]:
    """Basic lifespan context manager for FastAPI."""
    # Load environment variables from .env file
    load_dotenv()

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


def create_app(
    title: str = "FastAPI Application",
    version: str = get_version(),
    description: str = "Description of the FastAPI application",
    lifespan: asynccontextmanager = _lifespan,
) -> FastAPI:
    """Create a new instance of the application."""
    app = FastAPI(
        title=title,
        version=version,
        lifespan=lifespan,
        description=description,
        # Improve performance with ORJSONResponse
        default_response_class=ORJSONResponse,
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
