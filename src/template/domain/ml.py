from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import polars
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LinearLR
from torchmetrics.classification import MulticlassAccuracy

from template.core.ml import split_dataset, train_model
from template.domain.dataset import DEFAULT_COLUMN_NAME, Dataset

from template.domain.tokenizer import Tokenizer
from template.infrastructure.storage.adapter import PickleRepository
from template.infrastructure.storage.base import AbstractStorageInfra


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


class MLRepository(PickleRepository[NLPModel]):
    """Repository for persisting machine learning models as pickled files."""

    def __init__(self, infra_client: AbstractStorageInfra) -> None:
        super().__init__(
            infra_client,
            type_object=Tokenizer,
            prefix="models/",
        )


class MLService:
    """Service for managing datasets (preprocessed raw data)."""

    def __init__(self, repo: MLRepository):
        self.repo = repo

    async def get(self, identifier: str) -> NLPModel:
        """
        Get the path of a dataset by its identifier.

        Args:
            identifier (str): The identifier of the dataset.

        Returns:
            Path: The path of the dataset.
        """
        ml = await self.repo.get(identifier)

        if ml is None:
            raise FileNotFoundError(f"ML Model '{identifier}' does not exist.")

        return ml

    async def create(
        self,
        identifier: str,
        dataframe: polars.DataFrame,
        tokenizer: Tokenizer,
        device: str = "cuda",
        batch_size: int = 256,
        ratio_tests: float = 0.1,
        ratio_validation: float = 0.1,
        d_model: int = 256,
        d_hidden: int = 256,
        n_context: int = 10,
        lr: float = 1e-3,
        num_epochs: int = 1,
        scheduler_start_factor: float = 1.0,
        scheduler_end_factor: float = 1e-4,
        scheduler_total_iters: int = 0,
    ) -> NLPModel:
        """
        Create a dataset from the raw data.

        Args:
            identifier (str): The identifier for the dataset (file name without extension).
            dataframe (polars.DataFrame): The raw data as a polars DataFrame.
            tokenizer (Tokenizer): The tokenizer to use for the dataset.
            device (str): The device to use for training (default: "cuda").
            batch_size (int): The batch size for training (default: 256).
            ratio_tests (float): The ratio of the dataset to use for testing (default: 0.1).
            ratio_validation (float): The ratio of the dataset to use for validation (default: 0.1).
            d_model (int): The dimension of the model (default: 256).
            d_hidden (int): The dimension of the hidden layer (default: 256).
            n_context (int): The number of context tokens (default: 10).
            lr (float): The learning rate for the optimizer (default: 1e-3).
            num_epochs (int): The number of epochs to train the model (default: 1).
            scheduler_start_factor (float): The starting factor for the learning rate scheduler (default: 1.0).
            scheduler_end_factor (float): The ending factor for the learning rate scheduler (default: 1e-4).
            scheduler_total_iters (int): The total number of iterations for the learning rate scheduler (default: 0).

        Returns:
            Tokenizer: The created dataset (polars DataFrame).
        """
        # Fast failure if the identifier already exists (prevents unnecessary processing)
        if await self.repo.exists(identifier):
            raise FileExistsError(f"Tokenizer '{identifier}' already exists.")

        sentences = dataframe[DEFAULT_COLUMN_NAME].to_list()

        dataset = Dataset(
            sentences=sentences,
            type_tokenizer=type(tokenizer),
        )

        train_loader, test_loader, val_loader = split_dataset(
            dataset,
            batch_size=batch_size,
            ratio_tests=ratio_tests,
            ratio_validation=ratio_validation,
        )
        model = BengioMLP(
            d_model=d_model,
            d_hidden=d_hidden,
            n_context=n_context,
            tokenizer=tokenizer,
        ).to(device)

        metric = MulticlassAccuracy(num_classes=len(tokenizer.vocab))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            validation_loader=val_loader,
            optimizer=optimizer,
            metric=metric,
            criterion=criterion,
            device=device,
            num_epochs=num_epochs,
            start_factor=scheduler_start_factor,
            end_factor=scheduler_end_factor,
            total_iters=scheduler_total_iters,
            type_scheduler=LinearLR,
        )

        await self.repo.create(identifier, model)

        # Always move back to CPU before returning (safer for pickling)
        return model.cpu()

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
