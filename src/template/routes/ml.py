from __future__ import annotations

from typing import Annotated, Sequence

from fastapi import (
    APIRouter,
    Depends,
    status,
    Query,
)

from template.depends import inject_s3
from template.infrastructure.s3.base import S3Infrastructure
from template.repositories.ml import MLRepository
from template.services.ml import MLService


async def _get_service(s3_client: Annotated[S3Infrastructure, Depends(inject_s3)]) -> MLService:
    """Return a ready-to-use service instance."""
    repository = MLRepository(s3_client)
    return MLService(repository)


models_router = APIRouter(
    tags=["ML"],
)


@models_router.get("/", status_code=status.HTTP_200_OK)
async def route_list_models(
    service: Annotated[MLService, Depends(_get_service)],
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
) -> Sequence[str]:
    """List all models."""
    return await service.list()
