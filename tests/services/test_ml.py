from typing import AsyncIterator, Dict, List
from unittest.mock import patch

import polars
import pytest
import torch

from rename.application.ml import MLService, IntegrityError, NotFoundException, ParameterException
from rename.domain.ml import BengioMLP
from rename.domain.tokenizer import CharTokenizer
from rename.infrastructure.database.adapter import InMemorySQLiteDatabaseInfra
from rename.infrastructure.repositories.ml import MLMetaRepository, MLBlobRepository
from rename.infrastructure.storage.local import InMemoryStorageInfra


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


class DummyTokenizer:
    """Tiny character-level tokenizer suitable for unit tests."""

    sos_index: int = 0
    eos_index: int = 1

    def __init__(self) -> None:
        self._idx_to_char: Dict[int, str] = {
            0: "<s>",
            1: "</s>",
            2: "A",
            3: "B",
            4: "C",
        }
        self._char_to_idx: Dict[str, int] = {v: k for k, v in self._idx_to_char.items()}
        self.vocab: List[str] = list(self._idx_to_char.values())

    # --------------------------------------------------------------------- #
    # public helpers                                                        #
    # --------------------------------------------------------------------- #
    def encode(self, text: str) -> List[int]:
        """Return a list of token indices corresponding to the input text."""
        return [self._char_to_idx[ch] for ch in text if ch in self._char_to_idx]

    def decode(self, tokens: List[int]) -> List[str]:
        """Return a list of characters from token indices."""
        return [self._idx_to_char[tok] for tok in tokens]


class DummyModel(torch.nn.Module):
    """Stateless model that returns deterministic ascending logits."""

    def __init__(self, tokenizer: DummyTokenizer, n_context: int = 8) -> None:
        super().__init__()
        self.tokenizer: DummyTokenizer = tokenizer
        self.n_context: int = n_context
        self._device: torch.device = torch.device("cpu")
        self._vocab: int = len(tokenizer.vocab)

    # ------------------------------------------------------------------ #
    # required proxy properties                                          #
    # ------------------------------------------------------------------ #
    @property
    def device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------ #
    # torch.nn.Module interface                                          #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return always same logits for given input."""
        return torch.tensor(
            [[float(i + j) for i in range(self._vocab)] for j in range(x.shape[1])], device=self._device
        ).unsqueeze(0)


def test_filter_logits_temperature() -> None:
    """Verify temperature scaling divides logits."""
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    out = MLService._filter_logits(logits, temperature=2.0)
    assert torch.allclose(out, logits / 2.0)


def test_filter_logits_top_k() -> None:
    """Verify top-k keeps only k largest values."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = MLService._filter_logits(logits, top_k=2)
    expected = torch.tensor([[-torch.inf, -torch.inf, 3.0, 4.0]])
    assert torch.equal(out, expected)


def test_filter_logits_top_p() -> None:
    """Verify nucleus filtering masks tokens beyond cumulative prob threshold."""
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    out = MLService._filter_logits(logits, top_p=0.6)

    # Only the most probable token should remain finite.
    keep_mask = torch.isfinite(out)
    assert keep_mask.sum() == 1
    assert keep_mask[0, 0]  # highest logit kept


@pytest.mark.asyncio
async def test_generate_returns_expected_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """generate should concatenate decoded tokens until EOS."""
    tokenizer = DummyTokenizer()
    model = DummyModel(tokenizer)
    service = MLService(repo_ml=None, blob_ml=None)

    async def fake_stream(
        **_: object,
    ) -> AsyncIterator[int]:
        for tok in (2, 3, 1):  # 1 is EOS, will not be yielded
            yield tok

    monkeypatch.setattr(service, "stream", fake_stream, raising=True)
    result = await service.generate(blob=model)
    assert result == "AB"


@pytest.mark.asyncio
async def test_stream_parameter_validation() -> None:
    """stream should reject simultaneous top-k and top-p usage."""
    tokenizer = DummyTokenizer()
    model = DummyModel(tokenizer)
    service = MLService(repo_ml=None, blob_ml=None)

    with pytest.raises(ParameterException):
        _ = [tok async for tok in service.stream(model, top_k=1, top_p=0.9)]


@pytest.mark.asyncio
async def test_stream_emits_until_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stream should yield tokens until an EOS is sampled (EOS not yielded)."""
    tokenizer = DummyTokenizer()
    model = DummyModel(tokenizer)
    service = MLService(repo_ml=None, blob_ml=None)

    # Pre-determine sampled tokens for deterministic behaviour.
    tokens_to_emit = [2, 3, 1]  # EOS = 1

    def fake_multinomial(
        probs: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:  # noqa: D401
        """Return predetermined tokens in FIFO order."""
        token = tokens_to_emit.pop(0)
        return torch.tensor([[token]])

    monkeypatch.setattr(torch, "multinomial", fake_multinomial, raising=True)

    out: List[int] = [tok async for tok in service.stream(model, max_len=5)]
    assert out == [2, 3]  # EOS not included
    assert len(out) <= 5


async def test_stream_with_prompt() -> None:
    """stream should yield tokens starting from a prompt."""
    tokenizer = DummyTokenizer()
    model = DummyModel(tokenizer)
    service = MLService(repo_ml=None, blob_ml=None)
    prompt = "A"

    async for tok in service.stream(
        blob=model,
        prompt=prompt,
        max_len=5,
    ):
        assert tok in [i for i in range(0, 5)]
