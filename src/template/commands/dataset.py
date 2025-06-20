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


async def get_service_dataset():  # noqa
    """Get the dataset service."""
    from template.services.dataset import DatasetService
    from template.repositories.dataset import DatasetRepository
    from template.settings import get_settings
    from template.settings import get_storage_infra

    settings = get_settings()

    infra_storage = get_storage_infra(settings=settings)
    repo = DatasetRepository(infra_storage=infra_storage)
    return DatasetService(repo=repo)


@cli_dataset.command(name="get")
async def get_dataset(identifier: str = typer.Argument(..., help="Dataset identifier to get")) -> pl.DataFrame:
    """Get a dataset by its identifier."""
    service = await get_service_dataset()
    dataset = await service.get(identifier)
    typer.echo(f"Dataset '{identifier}':")
    typer.echo(dataset)
    return dataset


@cli_dataset.command(name="create")
async def create_dataset(
    raw_data: str = typer.Argument(..., help="Raw data identifier (raw data folder)"),
    identifier: str = typer.Argument("default", help="Dataset identifier to create"),
) -> pl.DataFrame:
    """Create a dataset from the raw data."""

    path = RAW_DATA_PATH / f"{raw_data}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Raw data file '{path}' does not exist.")

    content = path.read_text(encoding="utf-8")

    service = await get_service_dataset()
    dataset = await service.create(identifier, content)
    typer.echo(f"Dataset '{identifier}' created successfully.")
    return dataset


@cli_dataset.command(name="delete")
async def delete_dataset(
    identifier: str = typer.Argument(..., help="Dataset identifier to delete"),
):
    """Delete a dataset by its identifier."""

    service = await get_service_dataset()
    await service.delete(identifier)
    typer.echo(f"Dataset '{identifier}' deleted successfully.")


@cli_dataset.command(name="list")
async def list_datasets():
    """List all datasets in the repository."""

    service = await get_service_dataset()
    datasets = await service.list()

    if not datasets:
        typer.echo("No datasets found.")
        return

    typer.echo("Available datasets:")
    for dataset in datasets:
        typer.echo(f"- {dataset}")
