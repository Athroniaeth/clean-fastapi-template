from typing import List

import polars

from template.repositories.dataset import DatasetRepository


def _preprocess(dataset: str) -> list[str]:
    """
    Preprocess the dataset.

    Args:
        dataset (str): Raw dataset as a string.

    Returns:
        list[str]: Preprocessed list of sentences.
    """
    dataset = dataset.replace("<SOS>", "\n")  # IDK why, there are some "<SOS>" in the file
    sentences = dataset.split("\n")
    sentences = [sentence for sentence in sentences if len(sentence) > 3]
    sentences = [sentence.strip() for sentence in sentences]

    return sentences


DEFAULT_COLUMN_NAME = "text"


class DatasetService:
    """Service for managing datasets (preprocessed raw data)."""

    def __init__(self, repo: DatasetRepository):
        self.repo = repo

    async def get(self, identifier: str) -> polars.DataFrame:
        """
        Get the path of a dataset by its identifier.

        Args:
            identifier (str): The identifier of the dataset.

        Returns:
            Path: The path of the dataset.
        """
        df = await self.repo.get(identifier)

        if df is None:
            raise FileNotFoundError(f"Dataset '{identifier}' does not exist.")

        return df

    async def create(
        self,
        identifier: str,
        text: str,
    ) -> polars.DataFrame:
        """
        Create a dataset from the raw data.

        Args:
            identifier (str): The identifier for the dataset (file name without extension).
            text (str): The raw data as a string.

        Returns:
            pl.DataFrame: The created dataset (polars DataFrame).
        """
        # Fast failure if the identifier already exists (prevents unnecessary processing)
        if await self.repo.exists(identifier):
            raise FileExistsError(f"Dataset '{identifier}' already exists.")

        dataset = _preprocess(text)
        dataset = polars.DataFrame({DEFAULT_COLUMN_NAME: dataset})
        await self.repo.create(identifier, dataset)
        return dataset

    async def delete(self, identifier: str) -> None:
        """
        Delete a dataset by its identifier.

        Args:
            identifier (str): The identifier of the dataset.
        """
        if not await self.repo.delete(identifier):
            raise FileNotFoundError(f"Dataset '{identifier}' does not exist.")

    async def list(self) -> List[str]:
        """
        List all datasets in the repository.

        Returns:
            list[str]: A list of dataset identifiers (file names without extension).
        """
        return await self.repo.list()
