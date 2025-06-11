from contextlib import asynccontextmanager
from typing import Optional

import typer

from template.core.cli import AsyncTyper

cli_keys = AsyncTyper(
    name="keys",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Manage API keys for accessing the API.",
)


@asynccontextmanager
async def get_service():
    """Helper to retrieve a service instance with an active DB session."""

    from template.infrastructure.sql.base import get_db
    from template.repositories.api_keys import APIKeyRepository
    from template.services.api_keys import APIKeyService

    from template.settings import get_settings

    settings = get_settings()

    async with get_db(settings.database_url) as session:
        repo = APIKeyRepository(session)
        yield APIKeyService(repo)
        await session.close()


@cli_keys.command("create")
async def create_key(
    name: str = typer.Option(..., help="Name of the API key."),
    description: Optional[str] = typer.Option(None, help="Description of the API key."),
    is_active: bool = typer.Option(True, help="Whether the key should be active."),
):
    """
    Create a new API key and display its one-time plain key.

    Args:
        name (str): The name for the new key.
        description (Optional[str]): An optional description.
        is_active (bool): Whether the key is active upon creation.
    """

    from template.schemas.api_keys import APIKeyCreateSchema

    async with get_service() as service:
        schema = APIKeyCreateSchema(name=name, description=description, is_active=is_active)
        created_key = await service.create(schema)
        typer.echo(f"✅ Key created: ID={created_key.id}, Name={created_key.name}")
        typer.echo(f"🔑 Plain key (save it now): {created_key.plain_key}")


@cli_keys.command("delete")
async def delete_key(
    key_id: int = typer.Argument(..., help="ID of the API key to delete."),
):
    """
    Delete an API key by its ID.

    Args:
        key_id (int): The identifier of the key to delete.
    """

    async with get_service() as service:
        await service.delete(key_id)
        typer.echo(f"🗑️ Key with ID={key_id} deleted successfully.")


@cli_keys.command("list")
async def list_keys(
    active_only: bool = typer.Option(False, help="Show only active API keys."),
    skip: int = typer.Option(0, help="Number of records to skip."),
    limit: int = typer.Option(100, help="Maximum number of keys to return."),
):
    """
    List API keys with optional filters.

    Args:
        active_only (bool): Whether to show only active keys.
        skip (int): How many records to skip.
        limit (int): Maximum number of keys to show.
    """

    async with get_service() as service:
        keys = await service.list_all(skip=skip, limit=limit, active_only=active_only)
        for key in keys:
            typer.echo(f"ID={key.id} | Name={key.name} | Active={key.is_active}")


@cli_keys.command("activate")
async def activate_key(
    key_id: int = typer.Argument(..., help="ID of the API key to activate."),
):
    """
    Activate a specific API key by ID.

    Args:
        key_id (int): The identifier of the key to activate.
    """

    async with get_service() as service:
        updated = await service.activate(key_id, active=True)
        typer.echo(f"✅ Key ID={updated.id} activated.")


@cli_keys.command("deactivate")
async def deactivate_key(
    key_id: int = typer.Argument(..., help="ID of the API key to deactivate."),
):
    """
    Deactivate a specific API key by ID.

    Args:
        key_id (int): The identifier of the key to deactivate.
    """

    async with get_service() as service:
        updated = await service.activate(key_id, active=False)
        typer.echo(f"🚫 Key ID={updated.id} deactivated.")
