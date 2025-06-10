from contextlib import asynccontextmanager
from typing import Optional

import typer

from template.core.cli import AsyncTyper

cli_users = AsyncTyper(
    name="users",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Manage users for accessing the application.",
)


@asynccontextmanager
async def get_service():
    """Helper to retrieve a service instance with an active DB session."""

    from template.infrastructure.database import get_db
    from template.repositories.users import UserRepository
    from template.services.users import UserService

    from template.settings import get_settings

    settings = get_settings()

    async with get_db(settings.database_url) as session:
        repo = UserRepository(session)
        yield UserService(repo)
        await session.close()


@cli_users.command("create")
async def create_user(
    username: str = typer.Option(..., help="Username for the new user."),
    password: str = typer.Option(..., help="Password for the new user.", prompt=True, hide_input=True),
):
    """
    Create a new user.

    Args:
        username (str): The username for the new user.
        password (str): The password for the new user (will be prompted securely).
    """

    from template.schemas.users import UserCreateSchema

    async with get_service() as service:
        schema = UserCreateSchema(username=username, raw_password=password)
        created_user = await service.create(schema)
        typer.echo(f"✅ User created: ID={created_user.id}, Username={created_user.username}")


@cli_users.command("delete")
async def delete_user(
    user_id: int = typer.Argument(..., help="ID of the user to delete."),
):
    """
    Delete a user by its ID.

    Args:
        user_id (int): The identifier of the user to delete.
    """

    async with get_service() as service:
        await service.delete(user_id)
        typer.echo(f"🗑️ User with ID={user_id} deleted successfully.")


@cli_users.command("list")
async def list_users(
    skip: int = typer.Option(0, help="Number of records to skip."),
    limit: int = typer.Option(100, help="Maximum number of users to return."),
):
    """
    List users with optional pagination.

    Args:
        skip (int): How many records to skip.
        limit (int): Maximum number of users to show.
    """

    async with get_service() as service:
        users = await service.list_all(skip=skip, limit=limit)
        for user in users:
            typer.echo(f"ID={user.id} | Username={user.username}")


@cli_users.command("get")
async def get_user(
    user_id: int = typer.Argument(..., help="ID of the user to retrieve."),
):
    """
    Get a specific user by ID.

    Args:
        user_id (int): The identifier of the user to retrieve.
    """

    async with get_service() as service:
        user = await service.get(user_id)
        typer.echo(f"ID={user.id} | Username={user.username}")


@cli_users.command("update")
async def update_user(
    user_id: int = typer.Argument(..., help="ID of the user to update."),
    username: Optional[str] = typer.Option(None, help="New username for the user."),
    password: Optional[str] = typer.Option(None, help="New password for the user."),
):
    """
    Update a user's information.

    Args:
        user_id (int): The identifier of the user to update.
        username (Optional[str]): New username (optional).
        password (Optional[str]): New password (optional, will be prompted if provided).
    """

    from template.schemas.users import UserUpdateSchema

    # If password flag is provided but no value, prompt for it securely
    if password is not None and password == "":
        password = typer.prompt("New password", hide_input=True)

    update_data = {}
    if username is not None:
        update_data["username"] = username
    if password is not None:
        update_data["raw_password"] = password

    if not update_data:
        typer.echo("❌ No fields provided to update.")
        return

    async with get_service() as service:
        schema = UserUpdateSchema(**update_data)
        updated_user = await service.update(user_id, schema)
        typer.echo(f"✅ User updated: ID={updated_user.id}, Username={updated_user.username}")


@cli_users.command("verify")
async def verify_password(
    username: str = typer.Option(..., help="Username to verify."),
    password: str = typer.Option(..., help="Password to verify.", prompt=True, hide_input=True),
):
    """
    Verify a user's password.

    Args:
        username (str): The username to verify.
        password (str): The password to verify (will be prompted securely).
    """

    async with get_service() as service:
        try:
            await service.verify_password(username, password)
            typer.echo(f"✅ Password verified for user: {username}")
        except Exception as e:
            typer.echo(f"❌ Password verification failed: {e}")


@cli_users.command("get-by-username")
async def get_user_by_username(
    username: str = typer.Argument(..., help="Username of the user to retrieve."),
):
    """
    Get a specific user by username.

    Args:
        username (str): The username of the user to retrieve.
    """

    async with get_service() as service:
        user = await service.get_by_username(username)
        typer.echo(f"ID={user.id} | Username={user.username}")
