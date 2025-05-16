from contextlib import asynccontextmanager

from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse

from template import get_version
from template.core.state import FastAPI, lifespan
from template.core.exceptions import APIException
from template.routers.index import index_router, api_router


def create_app(
    title: str = "FastAPI Application",
    version: str = get_version(),
    description: str = "Description of the FastAPI application",
    lifespan: asynccontextmanager = lifespan,
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
