from fastapi import APIRouter

from template.core.state import Request

index_router = APIRouter(tags=["Utils"])
api_router = APIRouter(tags=["API"], prefix="/api/v1")


@index_router.get("/")
async def root(request: Request):
    """Root endpoint of the application."""
    return {
        "name": request.state.title,
        "version": request.state.version,
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@index_router.get("/health")
async def health():
    """Health endpoint of the application."""
    return {"status": "ok"}
