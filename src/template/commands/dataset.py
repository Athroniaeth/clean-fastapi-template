import polars as pl
import typer

from template import RAW_DATA_PATH
from template.core.cli import AsyncTyper

cli_dataset = AsyncTyper(
    name="dataset",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Manage datasets (preprocessed raw data) for tokenizer/ml.",
)


async def _get_service():  # noqa
    """Get the dataset service."""
    from template.services.dataset import DatasetService
    from template.repositories.dataset import DatasetRepository
    from template.infrastructure.database import get_s3_client

    from template.settings import get_settings

    settings = get_settings()

    async with get_s3_client() as s3_client:
        repo = DatasetRepository(
            s3_client=s3_client,
            bucket=settings.s3_bucket,
        )

        return DatasetService(repo=repo)


@cli_dataset.command(name="get")
async def get_dataset(
    identifier: str = typer.Argument(..., help="Dataset identifier to get"),
) -> pl.DataFrame:
    """Get a dataset by its identifier."""

    from template.services.dataset import DatasetService

    service: DatasetService = await _get_service()
    dataset = await service.get(identifier)
    typer.echo(f"Dataset '{identifier}':")
    typer.echo(dataset.head(1))
    return dataset


@cli_dataset.command(name="create")
async def create_dataset(
    raw_data: str = typer.Argument(..., help="Raw data as a string"),
    identifier: str = typer.Argument("default", help="Dataset identifier to create"),
) -> pl.DataFrame:
    """Create a dataset from the raw data."""

    from template.services.dataset import DatasetService

    path = RAW_DATA_PATH / f"{raw_data}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Raw data file '{path}' does not exist.")

    content = path.read_text(encoding="utf-8")

    service: DatasetService = await _get_service()
    dataset = await service.create(identifier, content)
    typer.echo(f"Dataset '{identifier}' created successfully.")
    return dataset


@cli_dataset.command(name="delete")
async def delete_dataset(
    identifier: str = typer.Argument(..., help="Dataset identifier to delete"),
):
    """Delete a dataset by its identifier."""

    from template.services.dataset import DatasetService

    service: DatasetService = await _get_service()
    await service.delete(identifier)
    typer.echo(f"Dataset '{identifier}' deleted successfully.")


@cli_dataset.command(name="list")
async def list_datasets():
    """List all datasets in the repository."""

    from template.services.dataset import DatasetService

    service: DatasetService = await _get_service()
    datasets = await service.list()

    if not datasets:
        typer.echo("No datasets found.")
        return

    typer.echo("Available datasets:")
    for dataset in datasets:
        typer.echo(f"- {dataset}")
