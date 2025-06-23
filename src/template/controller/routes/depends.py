from typing import Annotated, Optional, AsyncIterator

from aiobotocore.client import AioBaseClient
from fastapi import Depends, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from template.core.improve import Request
from template.infrastructure.database.api_keys import APIKeyRepository
from template.application.api_keys import APIKeyService


async def inject_db(request: Request) -> AsyncIterator[AsyncSession]:
    """Get the database session."""
    async_session = request.state.async_session

    async with async_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def inject_s3(request: Request) -> AsyncIterator[AioBaseClient]:
    """Get the S3 session."""
    s3_session = request.state.infra_storage

    try:
        yield s3_session
    except Exception:
        raise
    finally:
        pass  # No explicit close needed for boto3 sessions


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    session: Annotated[AsyncSession, Depends(inject_db)],
    api_key: Optional[str] = Security(api_key_header),
) -> None:
    """
    Verify that the API key is valid and active.

    Args:
        session: The database session.
        api_key: The API key to verify.

    Raises:
        KeyInactive: If key is unknown or inactive.
    """
    repo = APIKeyRepository(session)
    service = APIKeyService(repo)
    await service.verify_key(api_key)
