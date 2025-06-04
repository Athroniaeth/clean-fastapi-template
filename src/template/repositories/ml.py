from __future__ import annotations

from aiobotocore.client import AioBaseClient

from template.domain.ml import NLPModel
from template.domain.tokenizer import Tokenizer
from template.repositories._pickle import PickleRepository


class MLRepository(PickleRepository[NLPModel]):
    """Repository for persisting machine learning models as pickled files."""

    def __init__(self, s3_client: AioBaseClient, bucket: str) -> None:
        super().__init__(s3_client, bucket, type_object=Tokenizer, prefix="ml_models/", extension="pickle")
