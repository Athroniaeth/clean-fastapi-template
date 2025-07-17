from typing import Optional, AsyncIterator

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader

from rename.application.api_keys import ApiKeyService
from rename.application.ml import MLService
from rename.api.core.improve import Request
from rename.infrastructure.database.base import AbstractDatabaseInfra
from rename.infrastructure.repositories.api_keys import APIKeyRepository
from rename.infrastructure.repositories.ml import MLMetaRepository, MLBlobRepository
from rename.infrastructure.storage.base import AbstractStorageInfra
from rename.settings import Settings

# FastAPI global key security scheme
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


async def inject_infra_database(request: Request) -> AsyncIterator[AbstractDatabaseInfra]:
    """Get the infra database dependency."""
    yield request.state.infra_database


async def inject_infra_storage(request: Request) -> AsyncIterator[AbstractStorageInfra]:
    """Get the infra storage dependency."""
    yield request.state.infra_storage


async def inject_settings(request: Request) -> AsyncIterator[Settings]:
    """Get the settings dependency."""
    yield request.state.settings


async def get_service_ml(
    infra_database: AbstractDatabaseInfra = Depends(inject_infra_database),
    infra_storage: AbstractStorageInfra = Depends(inject_infra_storage),
) -> MLService:
    """Return a ready-to-use service instance."""
    repo_ml = MLMetaRepository(infra_database)
    blob_ml = MLBlobRepository(infra_storage)
    service = MLService(
        repo_ml=repo_ml,
        blob_ml=blob_ml,
    )
    return service


async def get_service_api_keys(
    infra: AbstractDatabaseInfra = Depends(inject_infra_database),
) -> ApiKeyService:
    """Return a ready-to-use API key service instance."""
    repo = APIKeyRepository(infra)
    service = ApiKeyService(repo)
    return service


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
    service: ApiKeyService = Depends(get_service_api_keys),
) -> None:
    """
    Verify that the API key is valid and active.

    Args
        api_key: The API key to verify.
        infra: The database infrastructure dependency.

    Raises:
        KeyInactive: If key is unknown or inactive.
    """
    await service.verify_key(api_key)
