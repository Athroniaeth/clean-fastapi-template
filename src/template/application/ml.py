from __future__ import annotations

from datetime import datetime
from typing import List, Type

import polars
from template.domain.ml import BengioMLP, Model

import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torchmetrics.classification import MulticlassAccuracy
from template.core.ml import split_dataset, train_model
from template.domain.dataset import DEFAULT_COLUMN_NAME, Dataset
from template.domain.tokenizer import Tokenizer
from template.domain.ml import MLMeta
from template.infrastructure.repositories.ml import MLMetaRepository, AbstractModelBlob, MLBlobRepository


class MLService:
    """Service for managing datasets (preprocessed raw data)."""

    def __init__(self, repo_ml: MLMetaRepository, blob_ml: MLBlobRepository):
        self.repo = repo_ml
        self.blob = blob_ml

    async def get(self, id_: str) -> Model:
        """
        Get the path of a dataset by its identifier.

        Args:
            id_ (str): The identifier of the dataset.

        Returns:
            Path: The path of the dataset.
        """
        meta = await self.repo.get(id_)

        if meta is None:
            raise Exception(f"Model '{id_}' does not exist.")

        blob = await self.blob.get(id_)

        if blob is None:
            raise FileNotFoundError(f"Integrity error: Model '{id_}' exists in metadata but not in blob storage.")

        return Model(meta=meta, blob=blob)

    async def create(
        self,
        id_: str,
        version: str,
        d_model: int,
        d_hidden: int,
        n_context: int,
        tokenizer: Tokenizer,
        model: Type[AbstractModelBlob] = BengioMLP,
    ) -> Model:
        """
        Create a dataset from the raw data.

        Args:
            id_ (str): The identifier for the dataset (file name without extension).
            version (str): The version of the dataset.
            d_model (int): The dimension of the model.
            d_hidden (int): The dimension of the hidden layer.
            n_context (int): The context size for the model.
            tokenizer (Tokenizer): The tokenizer to use for the dataset.
            model (Type[AbstractModelBlob]): The model class to instantiate (default: BengioMLP).

        Returns:
            Tokenizer: The created dataset (polars DataFrame).
        """

        meta = MLMeta(
            id_=id_,
            version=version,
            created_at=datetime.now(),
        )
        blob = model(
            d_model=d_model,
            d_hidden=d_hidden,
            n_context=n_context,
            tokenizer=tokenizer,
        )

        await self.blob.create(id_, blob)
        await self.repo.create(meta)
        return Model(
            meta=meta,
            blob=blob,
        )

    async def delete(self, id_: str) -> None:
        """
        Delete a dataset by its identifier.

        Args:
            id_ (int): The identifier of the dataset.
        """
        # Check existence and integrity before deletion
        await self.get(id_)
        await self.blob.delete(id_)
        await self.repo.delete(id_)

    async def list(self) -> List[Model]:
        """
        List all datasets in the repository.

        Returns:
            list[str]: A list of dataset identifiers (file names without extension).
        """
        list_model = []
        generator = zip(
            await self.repo.list_all(),
            await self.blob.list_all(),
        )
        for meta, blob in generator:
            model = Model(meta=meta, blob=blob)
            list_model.append(model)

        return list_model

    async def list_ids(self) -> List[str]:
        """
        List all dataset identifiers in the repository.

        Returns:
            list[str]: A list of dataset identifiers (file names without extension).
        """
        return [meta.id_ for meta in await self.repo.list_all()]

    async def train(
        self,
        id_: str,
        dataframe: polars.DataFrame,
        device: str = "cuda",
        batch_size: int = 256,
        ratio_tests: float = 0.1,
        ratio_validation: float = 0.1,
        lr: float = 1e-3,
        num_epochs: int = 1,
        scheduler_start_factor: float = 1.0,
        scheduler_end_factor: float = 1e-4,
        scheduler_total_iters: int = 0,
    ) -> Model:
        """
        Create a dataset from the raw data.

        Args:
            id_ (str): The identifier for the dataset (file name without extension).
            dataframe (polars.DataFrame): The raw data as a polars DataFrame.
            device (str): The device to use for training (default: "cuda").
            batch_size (int): The batch size for training (default: 256).
            ratio_tests (float): The ratio of the dataset to use for testing (default: 0.1).
            ratio_validation (float): The ratio of the dataset to use for validation (default: 0.1).
            lr (float): The learning rate for the optimizer (default: 1e-3).
            num_epochs (int): The number of epochs to train the model (default: 1).
            scheduler_start_factor (float): The starting factor for the learning rate scheduler (default: 1.0).
            scheduler_end_factor (float): The ending factor for the learning rate scheduler (default: 1e-4).
            scheduler_total_iters (int): The total number of iterations for the learning rate scheduler (default: 0).

        Returns:
            Tokenizer: The created dataset (polars DataFrame).
        """
        model = await self.get(id_)
        meta, blob, tokenizer = (
            model.meta,
            model.blob,
            model.blob.tokenizer,
        )

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

        metric = MulticlassAccuracy(num_classes=len(tokenizer.vocab))
        optimizer = torch.optim.Adam(blob.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_model(
            model=blob,
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
        # Ensure the model is on CPU before saving
        blob = blob.cpu()

        await self.blob.update(id_, blob)
        await self.repo.update(meta)
        return Model(
            meta=meta,
            blob=blob,
        )
