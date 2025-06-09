import contextlib

from loguru import logger
from torch.utils.data import SubsetRandomSampler, DataLoader, Dataset
import statistics
from typing import Type

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

import numpy as np


def split_dataset(
    dataset: Dataset,
    batch_size: int = 256,
    shuffle_dataset: bool = True,
    random_seed: int = 42,
    ratio_tests: float = 0.2,
    ratio_validation: float = 0.2,
):
    """
    Sépare le dataset en trois parties : entraînement, tests et validation.

    Args:
        dataset (Dataset): Dataset PyTorch à splitter.
        batch_size (int): Taille des batchs. Defaults to 32.
        shuffle_dataset (bool): Mélange le dataset. Defaults to True.
        random_seed (int): Graine aléatoire pour le mélange. Defaults to 42.
        ratio_tests (float): Ratio du dataset pour les tests. Defaults to 0.1.
        ratio_validation (float): Ratio du dataset pour la validation. Defaults to 0.1.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Tuple contenant les DataLoaders pour l'entraînement, les tests et la validation.
    """
    ratio = round(1 - ratio_tests - ratio_validation, 2)
    print(f"Splitting dataset, {ratio=}, {ratio_tests=}, {ratio_validation=}")

    # Prépare une liste d'indices du dataset
    dataset_size = len(dataset)  # noqa
    indices = [i for i in range(dataset_size)]

    # Calcule les slices pour chaque partie du dataset
    split_tests = dataset_size // (1 / ratio_tests)
    split_validation = dataset_size // (1 / ratio_validation)
    split_tests, split_validation = int(split_tests), int(split_validation)

    slice_train = slice(split_tests + split_validation, None)
    slice_tests = slice(split_tests, split_tests + split_validation)
    slice_validation = slice(None, split_tests)

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Applique les slices aux indices
    train_indices = indices[slice_train]
    tests_indices = indices[slice_tests]
    validation_indices = indices[slice_validation]

    # Crée les samplers (sous-ensembles du dataset)
    train_sampler = SubsetRandomSampler(train_indices)
    tests_sampler = SubsetRandomSampler(tests_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    # Crée les DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    tests_loader = DataLoader(dataset, batch_size=batch_size, sampler=tests_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)

    return train_loader, tests_loader, validation_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    validation_loader: DataLoader,
    num_epochs: int = 20,
    device: str = "cuda",
    optimizer: Optimizer = torch.optim.Adam,
    metric: MulticlassAccuracy = MulticlassAccuracy,
    criterion: _Loss = nn.CrossEntropyLoss,
    scheduler: Type[LinearLR] = LinearLR,
    start_factor: float = 1.0,
    end_factor: float = 1e-4,
    total_iters: int = 0,
):
    # Move model to device
    model = model.to(device)
    logger.info(f"Device: {model.device}")

    # Initialize optimizer and scheduler
    scheduler = scheduler(
        optimizer=optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=total_iters,
    )

    metric_name = metric.__class__.__name__

    # Training loop
    with contextlib.suppress(KeyboardInterrupt):
        for epoch in range(num_epochs):
            model.train()
            pbar = tqdm(train_loader, total=len(train_loader), leave=False)

            list_loss = []
            for x, y in pbar:
                x = x.to(device=device)
                y = y.to(device=device)

                # Forward pass
                predictions = model(x)  # [B, L, V]

                # Reshape for loss calculation
                predictions_flat = predictions.view(-1, predictions.size(-1))  # [B*L, V]
                y_flat = y.view(-1)  # [B*L]

                # Calculate loss
                loss = criterion(predictions_flat, y_flat)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description(f"Epochs: {epoch + 1}/{num_epochs}  -  Loss: {loss.item():.1e}")
                list_loss.append(loss.item())

        # Step scheduler
        scheduler.step()
        tqdm.write(f"Learning rate: {scheduler.get_last_lr()[0]:.1e}")

        # Evaluate
        train_score = check_accuracy(train_loader, model, metric)
        tests_score = check_accuracy(test_loader, model, metric)

        avg_loss = statistics.mean(list_loss)
        tqdm.write(
            f"Epochs: {epoch + 1}/{num_epochs}  -  "
            f"Loss: {avg_loss:.1e}  -  "
            f"Train ({metric_name}): {train_score:.4f}  -  "
            f"Tests ({metric_name}) : {tests_score:.4f}"
        )

    # Final evaluation
    validation_score = check_accuracy(validation_loader, model, metric)

    logger.info(f"[Final train score]: {check_accuracy(train_loader, model, metric):.4f}")
    logger.info(f"[Final test score]: {check_accuracy(test_loader, model, metric):.4f}")
    logger.info(f"[Final validation score]: {validation_score:.4f}")


def _calcul_accuracy(predictions: torch.Tensor, labels: torch.Tensor, tolerance: float = 1e-2):
    # Calculer le ratio entre les prédictions et les valeurs réelles
    ratio = predictions / labels

    # S'assurer que les ratios sont toujours >= 1
    mask = ratio < 1
    ratio[mask] = 1 / ratio[mask]

    # Vérifier si les prédictions sont dans la tolérance
    correct = (ratio <= (1 + tolerance)).sum().item()
    total = predictions.size(0)

    return correct, total


def check_accuracy(loader: DataLoader, model: nn.Module, metric: Metric):
    model.eval()
    device = model.device
    metric = metric.to(device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            predictions = model(x)  # [B, L, V]

            # Reshape predictions and targets for the metric
            predictions = predictions.view(-1, predictions.size(-1))  # [B*L, V]
            y = y.view(-1)  # [B*L]

            # Update metric
            metric.update(predictions, y)

    # Compute and reset metric
    accuracy = metric.compute().item()
    metric.reset()
    model.train()
    return accuracy
