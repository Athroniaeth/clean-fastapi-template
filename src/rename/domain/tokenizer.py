from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Self, Sequence, List


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

    @property
    def eos_index(self) -> int:
        """Return the index of the end of sequence token."""
        return self.token_to_index[self.eos_token]

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
