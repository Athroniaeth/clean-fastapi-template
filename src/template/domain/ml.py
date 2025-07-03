from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import torch
from pydantic import BaseModel, Field
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

    class Config:
        from_attributes = True

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
        *args,
        **kwargs,
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

    def generate_city_name(
        self,
        start_tokens: str = "",
        max_length: int = 30,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        temperature = max(temperature, 1e-6)
        self.eval()
        device = self.device

        current_input = torch.tensor([[self.tokenizer.sos_index]], device=device)

        if start_tokens:
            start_tokens_encoded = self.tokenizer.encode(start_tokens)
            start_tokens_tensor = torch.tensor([start_tokens_encoded], device=device)
            current_input = torch.cat([current_input, start_tokens_tensor], dim=1)
            generated_indices = [self.tokenizer.sos_index] + start_tokens_encoded
        else:
            generated_indices = [self.tokenizer.sos_index]

        with torch.no_grad():
            for _ in range(max_length):
                logits = self(current_input)[:, -1, :]  # (batch_size, vocab_size)
                logits = logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(logits, top_k)
                    mask = torch.full_like(logits, float("-inf"))
                    logits = mask.scatter(1, top_k_indices, top_k_values)

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(probs, dim=-1)

                    sorted_mask = cumulative_probs > top_p
                    # Shift mask right to include at least one token
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = 0

                    sorted_logits[sorted_mask] = float("-inf")
                    logits = logits.scatter(1, sorted_indices, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                if next_token == self.tokenizer.token_to_index[self.tokenizer.eos_token]:
                    break

                generated_indices.append(next_token)

                next_token_tensor = torch.tensor([[next_token]], device=device)
                current_input = torch.cat([current_input, next_token_tensor], dim=1)

                if current_input.size(1) > self.n_context:
                    current_input = current_input[:, -self.n_context :]

        generated_chars = self.tokenizer.decode(generated_indices[1:])  # Skip SOS
        return "".join(generated_chars)


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


@dataclass
class Model:
    """Model class that extends ModelMeta with blob pytorch model.

    Attributes:
        meta (MLMeta): Metadata for the model.
        blob (torch.nn.Module): The PyTorch model instance.
    """

    meta: MLMeta = Field(..., description="Metadata for the model")
    blob: AbstractModelBlob = Field(..., description="The PyTorch model instance")
