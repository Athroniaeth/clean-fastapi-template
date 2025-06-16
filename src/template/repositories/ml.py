from __future__ import annotations


from template.domain.ml import NLPModel
from template.domain.tokenizer import Tokenizer
from template.infrastructure.storage.adapter import PickleRepository
from template.infrastructure.storage.base import AbstractStorageInfra


class MLRepository(PickleRepository[NLPModel]):
    """Repository for persisting machine learning models as pickled files."""

    def __init__(self, infra_client: AbstractStorageInfra) -> None:
        super().__init__(
            infra_client,
            type_object=Tokenizer,
            prefix="models/",
        )
