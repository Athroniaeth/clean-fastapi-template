from __future__ import annotations

from typing import Annotated, Sequence

from fastapi import (
    APIRouter,
    Depends,
    status,
    Query,
)
from sqlalchemy.ext.asyncio import AsyncSession

from template.infrastructure.database import inject_db
from template.repositories.api_keys import APIKeyRepository
from template.schemas.api_keys import (
    APIKeyCreateResponseSchema,
    APIKeyCreateSchema,
    APIKeyReadResponseSchema,
    APIKeyUpdateSchema,
)
from template.services.api_keys import APIKeyService


async def _get_service(session: Annotated[AsyncSession, Depends(inject_db)]) -> APIKeyService:
    """Return a ready-to-use service instance."""
    repository = APIKeyRepository(session)
    return APIKeyService(repository)


keys_router = APIRouter(
    prefix="/keys",
    tags=["API Keys"],
)


@keys_router.get("/{id_}", status_code=status.HTTP_200_OK)
async def get_api_key(
    id_: int,
    service: Annotated[APIKeyService, Depends(_get_service)],
) -> APIKeyReadResponseSchema:
    return await service.get(id_)


@keys_router.post(
    "/",
    response_model=APIKeyCreateResponseSchema,
    status_code=status.HTTP_201_CREATED,
)
async def create_api_key(
    payload: Annotated[APIKeyCreateSchema, Query(...)],
    service: Annotated[APIKeyService, Depends(_get_service)],
) -> APIKeyCreateResponseSchema:
    return await service.create(payload)


@keys_router.get("/", status_code=status.HTTP_200_OK)
async def list_api_keys(
    service: Annotated[APIKeyService, Depends(_get_service)],
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    active_only: bool = Query(False),
) -> Sequence[APIKeyReadResponseSchema]:
    return await service.list_all(
        skip=skip,
        limit=limit,
        active_only=active_only,
    )


@keys_router.delete(
    "/{id_}",
    status_code=status.HTTP_204_NO_CONTENT,
    # Add examples output
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Root endpoint of the application",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                    },
                }
            },
        },
    },
)
async def delete_api_key(
    id_: int,
    service: Annotated[APIKeyService, Depends(_get_service)],
):
    await service.delete(id_)


@keys_router.patch("/{id_}", status_code=status.HTTP_200_OK)
async def update_api_key(
    id_: int,
    payload: Annotated[APIKeyUpdateSchema, Query()],
    service: Annotated[APIKeyService, Depends(_get_service)],
) -> APIKeyReadResponseSchema:
    return await service.update(id_, payload)
