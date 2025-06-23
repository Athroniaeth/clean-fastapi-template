from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Self, Sequence, List, Type

import polars as pl

from template.domain.dataset import DEFAULT_COLUMN_NAME
from template.infrastructure.storage.adapter import PickleRepository
from template.infrastructure.storage.base import AbstractStorageInfra


@dataclass
class Tokenizer(ABC):
    """
    Abstract base class for tokenizers.

    Args:
        vocab (Set[str]): A set of unique tokens in the vocabulary.
        token_to_index (Dict[str, int]): A mapping from tokens to indices.
        index_to_token (Dict[int, str]): A mapping from indices to tokens.
    """

    vocab: Sequence[str]
    token_to_index: Dict[str, int]
    index_to_token: Dict[int, str]

    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"

    @property
    def sos_index(self) -> int:
        """Return the index of the start of sequence token."""
        return self.token_to_index[self.sos_token]

    @classmethod
    @abstractmethod
    def from_sentences(cls, sentences: Sequence[str]) -> Self:
        """Initialize the tokenizer with a text."""
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def _encode(self, token: str) -> int:
        """Encode a single token into an index."""
        ...

    @abstractmethod
    def encode(self, tokens: Sequence[str]) -> List[int]:
        """Encode a list of tokens into indices."""
        ...

    @abstractmethod
    def _decode(self, index: int) -> str:
        """Decode a single integer into a token."""
        ...

    @abstractmethod
    def decode(self, indices: Sequence[int]) -> List[str]:
        """Decode a list of indices into a string."""
        ...


@dataclass
class CharTokenizer(Tokenizer):
    """A simple character-level tokenizer."""

    @classmethod
    def from_sentences(cls, sentences: Sequence[str]) -> Self:
        """Initialize the tokenizer with a text."""
        vocab = [
            CharTokenizer.pad_token,
            CharTokenizer.unk_token,
            CharTokenizer.sos_token,
            CharTokenizer.eos_token,
            *sorted(set("".join(sentences))),
        ]

        token_to_index = {}
        index_to_token = {}

        for index, char in enumerate(vocab):
            token_to_index[char] = index
            index_to_token[index] = char

        return cls(
            vocab=vocab,
            token_to_index=token_to_index,
            index_to_token=index_to_token,
        )

    def _encode(self, token: str) -> int:
        """Encode a single token into an index."""
        return self.token_to_index.get(token, self.token_to_index[self.unk_token])

    def encode(self, tokens: Sequence[str]) -> List[int]:
        """Encode a list of tokens into indices."""
        return [self._encode(token) for token in tokens]

    def _decode(self, index: int) -> str:
        """Decode a single integer into a token."""
        return self.index_to_token.get(index, self.unk_token)

    def decode(self, indices: Sequence[int]) -> List[str]:
        """Decode a list of indices into a string."""
        return [self._decode(token) for token in indices]


class TokenizerRepository(PickleRepository[Tokenizer]):
    """Repository for persisting tokenizer objects as pickled files."""

    def __init__(self, infra_storage: AbstractStorageInfra) -> None:
        super().__init__(
            infra_storage,
            type_object=Tokenizer,
            prefix="tokenizers/",
        )


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
        return await self.repo.list()
