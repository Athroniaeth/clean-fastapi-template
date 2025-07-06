from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import torch
from pydantic import BaseModel, Field, ConfigDict
from torch import nn
from torch.nn import functional as F

from template.domain.tokenizer import Tokenizer


class MLMeta(BaseModel):
    """Metadata for NLP models

    Attributes:
        id_ (UUID): Unique identifier for the model.
        version (str): Version of the model.
        created_at (datetime): Timestamp when the model was created.
    """

    model_config = ConfigDict(from_attributes=True)

    id_: str = Field(..., description="Unique identifier (name) for the model")
    version: str = Field("1.0.0", description="Version of the model")
    created_at: datetime = Field(default_factory=datetime.now)


class AbstractModelBlob(ABC, nn.Module):
    """
    Base class for NLP models. All models should inherit from this class.
    It provides a common interface and properties for all NLP models.

    Attributes:
        tokenizer (Tokenizer): Tokenizer instance for handling text data.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        n_context: int,
        tokenizer: Tokenizer,
    ):
        """
        Initialize the NLP model with the given parameters.

        Args:
            tokenizer (Tokenizer): Tokenizer instance for handling text data.
        """
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_context = n_context
        self.tokenizer = tokenizer

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    @abstractmethod
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model given input indices."""
        raise NotImplementedError("Subclasses must implement forward method.")


class BengioMLP(AbstractModelBlob):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        n_context: int,
        tokenizer: Tokenizer,
    ):
        super().__init__(
            d_model=d_model,
            d_hidden=d_hidden,
            n_context=n_context,
            tokenizer=tokenizer,
        )

        self.len_vocab = len(tokenizer.vocab)
        self.embed = nn.Embedding(self.len_vocab, d_model)

        self.fc1 = nn.Linear(n_context * d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, self.len_vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        B, L = idx.shape

        # Pad the sequence on the left to create context for the initial tokens.
        # We need n_context - 1 padding tokens.
        padded_idx = F.pad(idx, (self.n_context - 1, 0), value=self.tokenizer.sos_index)

        # Create sliding windows of size `n_context`.
        # Unfold creates a view of the tensor without copying data, which is very efficient.
        context_idx = padded_idx.unfold(dimension=1, size=self.n_context, step=1)  # (B, L, n_context)

        # Unfold gives [..., embed(t-1), embed(t)], we can flip to match the original order.
        # Flip the context_idx to match the original order.
        context_idx = torch.flip(context_idx, dims=[-1])

        # Get embeddings for all context windows at once.
        embedded_context = self.embed(context_idx)  # (B, L, n_context, d_model)

        # Reshape the embeddings to match the linear layer's input.
        # We combine the context and embedding dimensions.
        # Shape: (B, L, n_context * d_model)
        x = embedded_context.view(B, L, self.n_context * self.d_model)

        # Apply the first linear layer and activation function.
        x = torch.tanh(self.fc1(x))  # (B, L, d_hidden)
        logits = self.fc2(x)  # (B, L, vocab_size)

        return logits


@dataclass
class ML:
    """Model class that extends ModelMeta with blob pytorch model.

    Attributes:
        meta (MLMeta): Metadata for the model.
        blob (torch.nn.Module): The PyTorch model instance.
    """

    meta: MLMeta = Field(..., description="Metadata for the model")
    blob: AbstractModelBlob = Field(..., description="The PyTorch model instance")
