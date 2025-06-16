from __future__ import annotations


from template.domain.tokenizer import Tokenizer
from template.infrastructure.storage.adapter import PickleRepository
from template.infrastructure.storage.base import AbstractStorageInfra


class TokenizerRepository(PickleRepository[Tokenizer]):
    """Repository for persisting tokenizer objects as pickled files."""

    def __init__(self, infra_storage: AbstractStorageInfra) -> None:
        super().__init__(
            infra_storage,
            type_object=Tokenizer,
            prefix="tokenizers/",
        )
