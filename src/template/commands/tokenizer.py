import typer

from template.commands.dataset import get_service_dataset
from template.core.cli import AsyncTyper
from template.infrastructure.s3.base import S3Infrastructure

cli_tokenizer = AsyncTyper(
    name="tokenizer",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Manage tokenizers for ML models.",
)


async def get_service_tokenizer():  # noqa
    """Get the tokenizer service."""
    from template.repositories.tokenizer import TokenizerRepository

    from template.services.tokenizer import TokenizerService
    from template.settings import get_settings

    settings = get_settings()

    s3_client = S3Infrastructure.from_settings(settings)
    repo = TokenizerRepository(s3_client=s3_client)
    return TokenizerService(repo=repo)


@cli_tokenizer.command(name="get")
async def get_tokenizer(
    identifier: str = typer.Argument(..., help="Tokenizer identifier to get"),
):
    """Get a tokenizer by its identifier."""
    from template.services.tokenizer import TokenizerService

    service: TokenizerService = await get_service_tokenizer()
    tokenizer = await service.get(identifier)
    typer.echo(f"Tokenizer '{identifier}':")
    typer.echo(tokenizer)
    return tokenizer


@cli_tokenizer.command(name="create")
async def create_tokenizer(
    dataset: str = typer.Argument(..., help="Raw data as a string"),
    identifier: str = typer.Argument("default", help="Tokenizer identifier to create"),
):
    """Create a tokenizer from the raw data."""

    service_dataset = await get_service_dataset()
    service_tokenizer = await get_service_tokenizer()

    dataset = await service_dataset.get(identifier=dataset)
    await service_tokenizer.create(identifier, dataset)
    typer.echo(f"Tokenizer '{identifier}' created successfully.")


@cli_tokenizer.command(name="delete")
async def delete_tokenizer(
    identifier: str = typer.Argument(..., help="Tokenizer identifier to delete"),
):
    """Delete a tokenizer by its identifier."""

    service = await get_service_tokenizer()
    await service.delete(identifier)
    typer.echo(f"Tokenizer '{identifier}' deleted successfully.")


@cli_tokenizer.command(name="list")
async def list_tokenizers():
    """List all tokenizers in the repository."""

    service = await get_service_tokenizer()
    tokenizers = await service.list()

    if not tokenizers:
        typer.echo("No tokenizers found.")
        return

    typer.echo("Available tokenizers:")
    for tokenizer in tokenizers:
        typer.echo(f"- {tokenizer}")
