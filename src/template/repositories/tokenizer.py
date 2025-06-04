from __future__ import annotations

from aiobotocore.client import AioBaseClient

from template.domain.tokenizer import Tokenizer
from template.infrastructure.s3 import PickleRepository


class TokenizerRepository(PickleRepository[Tokenizer]):
    """Repository for persisting tokenizer objects as pickled files."""

    def __init__(self, s3_client: AioBaseClient, bucket: str) -> None:
        super().__init__(
            s3_client,
            bucket,
            type_object=Tokenizer,
            prefix="tokenizers/",
            extension="pickle",
        )
