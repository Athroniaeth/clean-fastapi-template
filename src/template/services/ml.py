from typing import List

import polars
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torchmetrics.classification import MulticlassAccuracy

from template.domain.dataset import NLPDataset
from template.domain.ml import NLPModel, BengioMLP
from template.domain.tokenizer import Tokenizer
from template.repositories.ml import MLRepository
from template.services.dataset import DEFAULT_COLUMN_NAME
from template.core.ml import split_dataset, train_model


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

        dataset = NLPDataset(
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
