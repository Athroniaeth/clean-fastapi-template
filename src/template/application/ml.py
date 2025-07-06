from datetime import datetime
from typing import List, Type, AsyncIterator

import polars
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torchmetrics.classification import MulticlassAccuracy

from template.api.core import split_dataset, train_model
from template.domain.dataset import DEFAULT_COLUMN_NAME, Dataset
from template.domain.ml import BengioMLP, ML
from template.domain.ml import MLMeta
from template.domain.tokenizer import Tokenizer
from template.infrastructure.repositories.ml import MLMetaRepository, AbstractModelBlob, MLBlobRepository


class MLService:
    """Service for managing datasets (preprocessed raw data)."""

    def __init__(self, repo_ml: MLMetaRepository, blob_ml: MLBlobRepository):
        self.repo = repo_ml
        self.blob = blob_ml

    async def get(self, id_: str) -> ML:
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

        return ML(meta=meta, blob=blob)

    async def create(
        self,
        id_: str,
        version: str,
        d_model: int,
        d_hidden: int,
        n_context: int,
        tokenizer: Tokenizer,
        model: Type[AbstractModelBlob] = BengioMLP,
    ) -> ML:
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
        return ML(
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

    async def list(self) -> List[ML]:
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
            model = ML(meta=meta, blob=blob)
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
        device: str = "cpu",
        batch_size: int = 256,
        ratio_tests: float = 0.1,
        ratio_validation: float = 0.1,
        lr: float = 1e-3,
        num_epochs: int = 1,
        scheduler_start_factor: float = 1.0,
        scheduler_end_factor: float = 1e-4,
        scheduler_total_iters: int = 0,
    ) -> ML:
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
        return ML(
            meta=meta,
            blob=blob,
        )

    @staticmethod
    def _filter_logits(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Helper function to filter logits based on temperature, top-k, and top-p.

        Args:
            logits (torch.Tensor): The raw logits from the model.
            temperature (float): Temperature for scaling logits (default: 1.0).
            top_k (int): The number of top logits to keep (default: 0, which means no filtering).
            top_p (float): The cumulative probability threshold for nucleus filtering (default: 1.0, which means no filtering).

        Returns:
            torch.Tensor: The filtered logits after applying temperature scaling, top-k, and top-p filtering.
        """
        # Temperature scaling
        safe_temperature = max(temperature, 1e-6)
        scaled_logits = logits / safe_temperature

        # Top-k filtering
        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(scaled_logits, k=top_k)
            minus_inf_mask = torch.full_like(scaled_logits, float("-inf"))
            scaled_logits = minus_inf_mask.scatter(
                dim=1,
                index=top_k_indices,
                src=top_k_values,
            )

        # Nucleus (Top-p) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # All tokens with cumulative probability greater than top_p are masked
            over_p_mask = cumulative_probs > top_p

            # Keep at least the first token
            over_p_mask[..., 1:] = over_p_mask[..., :-1].clone()
            over_p_mask[..., 0] = 0

            sorted_logits[over_p_mask] = float("-inf")
            scaled_logits = scaled_logits.scatter(
                dim=1,
                index=sorted_indices,
                src=sorted_logits,
            )

        return scaled_logits

    async def generate(
        self,
        blob: AbstractModelBlob,
        prompt: str = "",
        top_k: int = 0,
        top_p: float = 1.0,
        max_length: int = 30,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate a text sequence using the specified model.

        Args:
            blob (AbstractModelBlob): The model blob to use for generation.
            prompt (str): Initial prompt to start the generation.
            top_k (int): The number of top logits to keep for filtering (default: 0, which means no filtering).
            top_p (float): The cumulative probability threshold for nucleus filtering (default: 1.0, which means no filtering).
            max_length (int): Maximum length of the generated sequence (default: 30).
            temperature (float): Temperature for controlling randomness in generation (default: 1.0).

        Yields:
            str: The generated text sequence.
        """
        tokenizer = blob.tokenizer
        eos_idx = blob.tokenizer.eos_index
        tokens = [blob.tokenizer.sos_index]  # Start with the SOS token

        async for next_token in self.stream(
            blob=blob,
            prompt=prompt,
            top_k=top_k,
            top_p=top_p,
            max_len=max_length,
            temperature=temperature,
        ):
            # Stop if EOS token is generated
            if next_token == eos_idx:
                break

            # Else add the token to the sequence
            tokens.append(next_token)

            # When finished, decode the tokens to characters
        decoded_chars = tokenizer.decode(tokens[1:])  # Skip the SOS token
        generated_name = "".join(decoded_chars)
        return generated_name

    @torch.no_grad()
    async def stream(
        self,
        blob: AbstractModelBlob,
        prompt: str = "",
        top_k: int = 0,
        top_p: float = 1.0,
        max_len: int = 30,
        temperature: float = 1.0,
    ) -> AsyncIterator[int]:
        """
        Generate a text sequence using the specified model.

        Args:
            blob (AbstractModelBlob): The model blob to use for generation.
            prompt (str): Initial prompt to start the generation.
            top_k (int): The number of top logits to keep for filtering (default: 0, which means no filtering).
            top_p (float): The cumulative probability threshold for nucleus filtering (default: 1.0, which means no filtering).
            max_len (int): Maximum length of the generated sequence (default: 30).
            temperature (float): Temperature for controlling randomness in generation (default: 1.0).

        Returns:
            str: The generated city name.
        """
        tokenizer = blob.tokenizer

        # Must have only top_p or top_k, not both
        have_top_k = top_k > 0
        have_top_p = top_p != 1.0

        if have_top_k and have_top_p:
            raise ValueError("You must choose either top_p or top_k, not both.")

        blob.eval()
        device = blob.device
        sos_idx = tokenizer.sos_index

        # Indicate to model that we start a new sequence
        tokens = [sos_idx]

        # Add the prompt to the tokens if provided
        if prompt:
            prompt_ids = tokenizer.encode(prompt)  # -> List[int]
            tokens.extend(prompt_ids)

        for _ in range(max_len):
            # Construct the context window from the last n_context tokens
            start_of_window = max(0, len(tokens) - blob.n_context)
            window_tokens = tokens[start_of_window:]  # sous-liste
            window_tensor = torch.tensor(
                [window_tokens],  # shape (1, window_len)
                dtype=torch.long,
                device=device,
            )

            # Start inference with the model
            model_output = blob(window_tensor)  # (1, window_len, vocab)
            last_position_logits = model_output[:, -1, :]  # (1, vocab)

            # Filter logits with temperature, top_k, and top_p
            filtered_logits = self._filter_logits(
                last_position_logits,
                temperature,
                top_k,
                top_p,
            )

            probs = torch.softmax(filtered_logits, dim=-1)  # (1, vocab)
            sampled_token_tensor = torch.multinomial(probs, num_samples=1)  # (1, 1)
            next_token: int = sampled_token_tensor.item()

            tokens.append(next_token)

            # If the next token is the end of sequence token, stop generating
            if next_token == tokenizer.eos_index:
                break
            yield next_token
