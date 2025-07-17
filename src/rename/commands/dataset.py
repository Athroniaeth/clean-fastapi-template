from typing import List, Optional

import typer

from rename import RAW_DATA_PATH
from rename.api.core.cli import AsyncTyper

cli_dataset = AsyncTyper(
    name="dataset",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Manage datasets (preprocessed raw data) for tokenizer/ml.",
)


async def get_service_dataset():  # noqa
    """Get the dataset service."""
    from rename.domain.dataset import DatasetService
    from rename.domain.dataset import DatasetRepository
    from rename.settings import get_settings
    from rename.settings import get_storage_infra

    settings = get_settings()

    infra_storage = get_storage_infra(settings=settings)
    repo = DatasetRepository(infra_storage=infra_storage)
    return DatasetService(repo=repo)


@cli_dataset.command(name="get")
async def get_dataset(identifier: str = typer.Argument(..., help="Dataset identifier to get")):
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
):
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


@cli_dataset.command(name="merge")
async def merge_datasets(
    dataset: List[str] = typer.Option(..., help="Comma-separated list of dataset identifiers to merge"),
    output: str = typer.Argument("merged_dataset", help="Output dataset identifier"),
    ratio: Optional[float] = typer.Option(
        None, help="Ratio of data to merge (default: 1.0, meaning 100% size of the smallest dataset)"
    ),
):
    """Merge multiple datasets into a new dataset."""
    service = await get_service_dataset()
    await service.merge(
        identifiers=dataset,
        output_id=output,
        ratio=ratio,
    )
    typer.echo(f"Datasets '{', '.join(dataset)}' merged into '{output}' successfully.")
