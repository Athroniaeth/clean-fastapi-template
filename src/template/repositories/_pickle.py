from __future__ import annotations

import logging
import pickle
from typing import Type, TypeVar, Generic

from aiobotocore.client import AioBaseClient

from template.repositories.s3 import AbstractS3Repository

T = TypeVar("T")

logger = logging.getLogger(__name__)


class PickleRepository(AbstractS3Repository, Generic[T]):
    """Repository in charge of persisting pickled objects to S3.

    This version expects the client to be injected (e.g. via FastAPI lifespan or DI).
    """

    def __init__(
        self,
        s3_client: AioBaseClient,
        bucket: str,
        type_object: Type[T],
        prefix: str = "",
        extension: str = "",
    ) -> None:
        super().__init__(
            s3_client,
            bucket,
            type_object,
            prefix=prefix,
            extension=extension,
        )

    def serialize(self, obj: T) -> bytes:
        """Convert obj to raw bytes ready for S3 upload."""
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize(self, payload: bytes) -> T:
        """Transform raw bytes downloaded from S3 back into an object of type T."""
        return pickle.loads(payload)
