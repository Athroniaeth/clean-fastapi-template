from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_session

from template.core.state import FastAPI, get_version, State
from template.core.exceptions import APIException
from template.infrastructure.database import get_db
from template.routes.index import index_router, api_router
from template.settings import get_settings


def create_app(
    title: str,
    version: str,
    description: str,
    session: AsyncSession,
) -> FastAPI:
    """Create a new instance of the application."""
    app = FastAPI(
        title=title,
        version=version,
        description=description,
        # Improve performance with ORJSONResponse
        default_response_class=ORJSONResponse,
    )

    app.state = State(
        session=session,
        title=app.title,
        version=app.version,
        description=app.description,
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


if __name__ == "__main__":
    settings = get_settings()

    # Create an async engine
    async_session = get_db(settings.database_url)

    # Create the FastAPI app
    app = create_app(
        title="Template FastAPI App",
        version=get_version(),
        description="A template FastAPI application with async capabilities.",
        session=async_session,
    )
