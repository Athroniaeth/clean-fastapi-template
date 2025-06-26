from typing import Annotated, Optional, AsyncIterator

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader

from template.application.api_keys import APIKeyService
from template.core.improve import Request
from template.infrastructure.database.base import AbstractDatabaseInfra
from template.infrastructure.repositories.api_keys import APIKeyRepository
from template.infrastructure.storage.base import AbstractStorageInfra


async def inject_infra_database(request: Request) -> AsyncIterator[AbstractDatabaseInfra]:
    """Get the infra database dependency."""
    yield request.state.infra_database


async def inject_infra_storage(request: Request) -> AsyncIterator[AbstractStorageInfra]:
    """Get the infra storage dependency."""
    yield request.state.infra_storage


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Annotated[Optional[str], Security(api_key_header)],
    infra: Annotated[AbstractDatabaseInfra, Depends(inject_infra_database)],
) -> None:
    """
    Verify that the API key is valid and active.

    Args
        api_key: The API key to verify.
        infra: The database infrastructure dependency.

    Raises:
        KeyInactive: If key is unknown or inactive.
    """
    repo = APIKeyRepository(infra)
    service = APIKeyService(repo)
    await service.verify_key(api_key)
