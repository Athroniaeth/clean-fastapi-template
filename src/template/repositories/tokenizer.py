from __future__ import annotations


from template.domain.tokenizer import Tokenizer
from template.infrastructure.s3.adapter import PickleRepository
from template.infrastructure.s3.base import S3Infrastructure


class TokenizerRepository(PickleRepository[Tokenizer]):
    """Repository for persisting tokenizer objects as pickled files."""

    def __init__(self, s3_client: S3Infrastructure) -> None:
        super().__init__(
            s3_client,
            type_object=Tokenizer,
            prefix="tokenizers/",
        )
