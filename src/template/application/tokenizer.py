from __future__ import annotations

from typing import Type, List

import polars as pl

from template.domain.tokenizer import Tokenizer, CharTokenizer
from template.infrastructure.repositories.tokenizer import TokenizerRepository


class TokenizerService:
    """Service for managing datasets (preprocessed raw data)."""

    def __init__(self, repo: TokenizerRepository):
        self.repo = repo

    async def get(self, identifier: str) -> Tokenizer:
        """
        Get the path of a dataset by its identifier.

        Args:
            identifier (str): The identifier of the dataset.

        Returns:
            Path: The path of the dataset.
        """
        df = await self.repo.get(identifier)

        if df is None:
            raise FileNotFoundError(f"Tokenizer '{identifier}' does not exist.")

        return df

    async def create(
        self,
        identifier: str,
        dataset: pl.DataFrame,
        class_: Type[Tokenizer] = CharTokenizer,
    ) -> Tokenizer:
        """
        Create a dataset from the raw data.

        Args:
            identifier (str): The identifier for the dataset (file name without extension).
            dataset (pl.DataFrame): The raw data as a polars DataFrame.
            class_ (Type[Tokenizer]): The tokenizer class to use for the dataset (default: CharTokenizer).

        Returns:
            Tokenizer: The created dataset (polars DataFrame).
        """
        from template.domain.dataset import DEFAULT_COLUMN_NAME

        # Fast failure if the identifier already exists (prevents unnecessary processing)
        if await self.repo.exists(identifier):
            raise FileExistsError(f"Tokenizer '{identifier}' already exists.")

        sentences = dataset[DEFAULT_COLUMN_NAME].to_list()
        tokenizer = class_.from_sentences(sentences=sentences)
        await self.repo.create(identifier, tokenizer)
        return tokenizer

    async def delete(self, identifier: str) -> None:
        """
        Delete a dataset by its identifier.

        Args:
            identifier (str): The identifier of the dataset.
        """
        if not await self.repo.delete(identifier):
            raise FileNotFoundError(f"Tokenizer '{identifier}' does not exist.")

    async def list(self) -> List[str]:
        """
        List all datasets in the repository.

        Returns:
            list[str]: A list of dataset identifiers (file names without extension).
        """
        return await self.repo.list_ids()
