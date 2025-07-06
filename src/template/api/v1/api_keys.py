from typing import Annotated, Sequence

from fastapi import (
    APIRouter,
    Depends,
    status,
    Query,
)

from template.domain.api_keys import ApiKey
from template.api.depends import inject_infra_database
from template.infrastructure.database.base import AbstractDatabaseInfra
from template.infrastructure.repositories.api_keys import APIKeyRepository
from template.application.api_keys import ApiKeyService
from template.api.schemas.api_keys import (
    APIKeyCreate,
    APIKeyRead,
    APIKeyUpdate,
    APIKeyCreateResponse,
)


async def _get_service(
    infra: Annotated[
        AbstractDatabaseInfra,
        Depends(inject_infra_database),
    ],
) -> ApiKeyService:
    """Return a ready-to-use service instance."""
    repository = APIKeyRepository(infra)
    return ApiKeyService(repository)


keys_router = APIRouter(
    prefix="/keys",
    tags=["API Keys"],
)


@keys_router.get("/{id_}", status_code=status.HTTP_200_OK)
async def get_api_key(
    id_: int,
    service: Annotated[ApiKeyService, Depends(_get_service)],
) -> APIKeyRead:
    """Retrieve an API key by its ID."""
    api_key = await service.get(id_)
    return APIKeyRead.model_validate(api_key)


@keys_router.post(
    "/",
    response_model=APIKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_api_key(
    payload: Annotated[APIKeyCreate, Query(...)],
    service: Annotated[ApiKeyService, Depends(_get_service)],
) -> APIKeyCreateResponse:
    """Create a new API key."""
    api_key = await service.create(
        name=payload.name,
        description=payload.description,
        is_active=payload.is_active,
    )
    return APIKeyCreateResponse.model_validate(api_key)


@keys_router.get("/", status_code=status.HTTP_200_OK)
async def list_api_keys(
    service: Annotated[ApiKeyService, Depends(_get_service)],
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=100),
    active_only: bool = Query(default=False),
) -> Sequence[APIKeyRead]:
    """List all API keys with optional pagination and active‐only filtering."""
    api_keys = await service.list_all(
        skip=skip,
        limit=limit,
        active_only=active_only,
    )
    return [APIKeyRead.model_validate(api_key) for api_key in api_keys]


@keys_router.delete(
    "/{id_}",
    status_code=status.HTTP_200_OK,
)
async def delete_api_key(
    id_: int,
    service: Annotated[ApiKeyService, Depends(_get_service)],
) -> APIKeyRead:
    """Delete an API key by its ID."""
    api_key = await service.get(id_)
    await service.delete(api_key)
    return APIKeyRead.model_validate(api_key)


@keys_router.patch("/{id_}", status_code=status.HTTP_200_OK)
async def update_api_key(
    payload: Annotated[APIKeyUpdate, Query()],
    service: Annotated[ApiKeyService, Depends(_get_service)],
) -> APIKeyRead:
    """Update an API key by its ID."""
    api_key = ApiKey.model_validate(payload)
    api_key = await service.update(api_key)
    return APIKeyRead.model_validate(api_key)
