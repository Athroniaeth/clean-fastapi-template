from typing import List, Optional

import polars as pl
from loguru import logger

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

    async def get(self, identifier: str) -> pl.DataFrame:
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
    ) -> pl.DataFrame:
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
        dataset = pl.DataFrame({DEFAULT_COLUMN_NAME: dataset})
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

    async def merge(self, identifiers: List[str], output_id: str, ratio: Optional[float] = None) -> pl.DataFrame:
        """
        Merge multiple datasets into a single dataset.

        Args:
            identifiers (List[str]): List of dataset identifiers to merge.
            output_id (str): The identifier for the output dataset.
            ratio (Optional[float]): Ratio of data to merge (default: None, meaning 100% size of the smallest dataset).

        Returns:
            pl.DataFrame: The merged dataset (polars DataFrame).
        """
        dfs = []
        if not identifiers:
            raise ValueError("No identifiers provided for merging datasets.")

        if await self.repo.exists(output_id):
            raise FileExistsError(f"Output dataset '{output_id}' already exists.")

        # Merge all given datasets
        for identifier in identifiers:
            df = await self.get(identifier)
            dfs.append(df)

        if ratio:
            _dfs = []
            min_rows = min(len(df) for df in dfs)
            min_rows_ratio = min(len(df) for df in dfs) * ratio
            logger.debug(f"Normalizing datasets to have the same number of rows. {ratio=}")

            for id, df in zip(identifiers, dfs):
                if len(df) > min_rows_ratio:
                    sampled_df = df.sample(n=min_rows_ratio, shuffle=True)
                    _dfs.append(sampled_df)
                else:
                    sampled_df = df.sample(n=len(df), shuffle=True)
                    _dfs.append(sampled_df)

                logger.debug(f"- '{id}' dataset sampled to {sampled_df.shape[0]} rows")

            dfs = _dfs
            total_rows = sum(len(df) for df in dfs)
            logger.debug(
                f"Minimum number of rows: ~{min_rows}, pass to {total_rows} rows ({min_rows} to {min_rows_ratio:.0f} rows per dataset)"
            )
        else:
            total_rows = sum(len(df) for df in dfs)
            logger.debug(f"Merging datasets, total rows: {total_rows}")

        merged_df = pl.concat(dfs)

        # Remove duplicates based on the "text" column
        merged_df = merged_df.unique(subset=["text"])

        await self.repo.create(output_id, merged_df)
        return merged_df
