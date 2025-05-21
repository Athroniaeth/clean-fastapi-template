from fastapi import APIRouter, Depends

from template.core.state import Request
from template.routes.api_keys import keys_router
from template.depends import verify_api_key

index_router = APIRouter(tags=["Utils"])
api_router = APIRouter(tags=["API"], prefix="/api/v1")
api_router.include_router(keys_router)


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


@index_router.get("/protected", dependencies=[Depends(verify_api_key)])
async def protected():
    """Protected endpoint of the application."""
    return {"status": "ok"}
