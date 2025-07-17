from rename.domain.tokenizer import Tokenizer
from rename.infrastructure.storage.local import PickleRepository
from rename.infrastructure.storage.base import AbstractStorageInfra


class TokenizerRepository(PickleRepository[Tokenizer]):
    """Repository for persisting tokenizer objects as pickled files."""

    def __init__(self, infra_storage: AbstractStorageInfra) -> None:
        super().__init__(
            infra_storage,
            type_object=Tokenizer,
            prefix="tokenizers/",
        )
