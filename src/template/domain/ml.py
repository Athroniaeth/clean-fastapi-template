from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F

from template.domain.tokenizer import Tokenizer


class NLPModel(ABC, nn.Module):
    """
    Base class for NLP models. All models should inherit from this class.
    It provides a common interface and properties for all NLP models.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        *args,
        **kwargs,
    ):
        """
        Initialize the NLP model with the given parameters.

        Args:
            tokenizer (Tokenizer): Tokenizer instance for handling text data.
        """
        super().__init__()
        self.tokenizer = tokenizer

    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device

    @abstractmethod
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model given input indices."""
        raise NotImplementedError("Subclasses must implement forward method.")


class BengioMLP(NLPModel):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        n_context: int,
        tokenizer: Tokenizer,
    ):
        super().__init__(tokenizer)

        self.n_context = n_context

        self.len_vocab = len(tokenizer.vocab)
        self.embed = nn.Embedding(self.len_vocab, d_model)

        self.fc1 = nn.Linear(n_context * d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, self.len_vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # Create a list to store context embeddings
        context_tokens = []

        # For each position in the context window
        for i in range(self.n_context):
            if i == 0:
                # Current token
                context_tokens.append(self.embed(idx))  # (B, L, d_model)
            else:
                # Get previous tokens by shifting
                shifted_idx = torch.roll(idx, shifts=i, dims=1)
                # Replace the first i positions with SOS token
                shifted_idx[:, :i] = self.tokenizer.sos_index
                context_tokens.append(self.embed(shifted_idx))

        # Concatenate all embeddings along the last dimension
        x = torch.cat(context_tokens, dim=-1)  # (B, L, n_context*d_model)

        # Process through MLP
        x = F.tanh(self.fc1(x))  # (B, L, d_hidden)
        logits = self.fc2(x)  # (B, L, vocab_size)

        return logits
