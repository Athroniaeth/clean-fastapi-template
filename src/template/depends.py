from typing import Annotated, Optional, AsyncIterator

from fastapi import Depends, Security
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from template.repositories.api_keys import APIKeyRepository
from template.services.api_keys import api_key_header, APIKeyService


async def inject_db(request: Request) -> AsyncIterator[AsyncSession]:
    """Get the database session."""
    async_session = request.state.session

    async with async_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


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
