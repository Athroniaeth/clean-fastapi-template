from unittest.mock import patch

import polars
import pytest

from template.application.ml import MLService, IntegrityError, NotFoundException
from template.domain.ml import BengioMLP
from template.domain.tokenizer import CharTokenizer
from template.infrastructure.database.adapter import InMemorySQLiteDatabaseInfra
from template.infrastructure.repositories.ml import MLMetaRepository, MLBlobRepository
from template.infrastructure.storage.local import InMemoryStorageInfra


@pytest.fixture(scope="function")
async def service(infra_database: InMemorySQLiteDatabaseInfra, infra_storage: InMemoryStorageInfra):
    repo_ml = MLMetaRepository(infra_database=infra_database)
    blob_ml = MLBlobRepository(infra_storage=infra_storage)
    service = MLService(repo_ml=repo_ml, blob_ml=blob_ml)
    return service


async def test_create_and_get_model(service: MLService):
    """Test creating and retrieving a model."""

    # Create the model
    model = await service.create(
        id_="test",
        version="1.0.0",
        d_model=128,
        d_hidden=64,
        n_context=10,
        tokenizer=CharTokenizer.from_sentences(["This is a test sentence."]),
        model=BengioMLP,
    )

    # Retrieve the model
    retrieved_model = await service.get("test")

    assert retrieved_model.meta.id_ == "test"
    assert retrieved_model.meta.version == "1.0.0"
    assert retrieved_model.meta.created_at == model.meta.created_at


async def test_create_and_delete_model(service: MLService):
    """Test creating and deleting a model."""
    # Create the model
    await service.create(
        id_="delete_test",
        version="1.0.0",
        d_model=128,
        d_hidden=64,
        n_context=10,
        tokenizer=CharTokenizer.from_sentences(["This is a test sentence."]),
        model=BengioMLP,
    )

    # Delete the model
    await service.delete("delete_test")

    # Try to retrieve the deleted model
    with pytest.raises(Exception):
        await service.get("delete_test")


async def test_list_models(service: MLService):
    """Test listing created models."""
    # Create multiple models
    await service.create(
        id_="model1",
        version="1.0.0",
        d_model=128,
        d_hidden=64,
        n_context=10,
        tokenizer=CharTokenizer.from_sentences(["This is a test sentence."]),
        model=BengioMLP,
    )
    await service.create(
        id_="model2",
        version="1.0.0",
        d_model=128,
        d_hidden=64,
        n_context=10,
        tokenizer=CharTokenizer.from_sentences(["This is another test sentence."]),
        model=BengioMLP,
    )

    models = await service.list_all()

    assert len(models) == 2
    assert any(model.meta.id_ == "model1" for model in models)
    assert any(model.meta.id_ == "model2" for model in models)


async def test_list_models_id(service: MLService):
    """Test listing identifiers created models."""
    # Create multiple models
    await service.create(
        id_="model1",
        version="1.0.0",
        d_model=128,
        d_hidden=64,
        n_context=10,
        tokenizer=CharTokenizer.from_sentences(["This is a test sentence."]),
        model=BengioMLP,
    )
    await service.create(
        id_="model2",
        version="1.0.0",
        d_model=128,
        d_hidden=64,
        n_context=10,
        tokenizer=CharTokenizer.from_sentences(["This is another test sentence."]),
        model=BengioMLP,
    )

    ids = await service.list_ids()

    assert len(ids) == 2
    assert ids == ["model1", "model2"]


async def test_train_model(service: MLService):
    """Test training a model.

    Mock template.core.ml.train_model function to avoid actual training.
    """
    # Todo: This test is too slow
    pytest.skip("This test is too slow, please reactivate it after mocking it.")
    dataframe = polars.DataFrame({"text": ["This is a test sentence."]})

    with patch("template.application.ml.train_model") as mock_train:
        await service.create(
            id_="train_test",
            version="1.0.0",
            d_model=128,
            d_hidden=64,
            n_context=10,
            tokenizer=CharTokenizer.from_sentences(["This is a test sentence."]),
            model=BengioMLP,
        )

        # Call the train method
        await service.train(
            "train_test",
            dataframe,
            device="cpu",
            batch_size=2,
            ratio_tests=0.1,
            ratio_validation=0.1,
            lr=1e-3,
            num_epochs=1,
            scheduler_start_factor=1.0,
            scheduler_end_factor=1e-4,
            scheduler_total_iters=0,
        )

        # Assert that the train_model function was called
        mock_train.assert_called_once()


async def test_get_non_existent_model(service: MLService):
    """Test getting a non-existent model raises an exception."""
    with pytest.raises(NotFoundException):
        await service.get("non_existent")


async def test_delete_non_existent_model(service: MLService):
    """Test deleting a non-existent model raises an exception."""
    with pytest.raises(NotFoundException):
        await service.delete("non_existent")


async def test_create_duplicate_model(service: MLService):
    """Test creating a model with the same ID raises an exception."""
    await service.create(
        id_="duplicate",
        version="1.0.0",
        d_model=128,
        d_hidden=64,
        n_context=10,
        tokenizer=CharTokenizer.from_sentences(["This is a test sentence."]),
        model=BengioMLP,
    )

    with pytest.raises(Exception):
        await service.create(
            id_="duplicate",
            version="1.0.0",
            d_model=128,
            d_hidden=64,
            n_context=10,
            tokenizer=CharTokenizer.from_sentences(["This is a test sentence."]),
            model=BengioMLP,
        )


async def test_integrity_meta_not_found(service: MLService):
    """Test integrity error when model blob exists but metadata does not."""
    await service.create(
        id_="integrity_meta_test",
        version="1.0.0",
        d_model=128,
        d_hidden=64,
        n_context=10,
        tokenizer=CharTokenizer.from_sentences(["This is a test sentence."]),
        model=BengioMLP,
    )

    # Simulate metadata not being created
    await service.repo_meta.delete("integrity_meta_test")

    with pytest.raises(Exception):
        await service.get("integrity_meta_test")


async def test_integrity_blob_not_found(service: MLService):
    """Test integrity error when model metadata exists but blob does not."""
    await service.create(
        id_="integrity_test",
        version="1.0.0",
        d_model=128,
        d_hidden=64,
        n_context=10,
        tokenizer=CharTokenizer.from_sentences(["This is a test sentence."]),
        model=BengioMLP,
    )

    # Simulate blob not being created
    await service.repo_blob.delete("integrity_test")

    with pytest.raises(IntegrityError):
        await service.get("integrity_test")
