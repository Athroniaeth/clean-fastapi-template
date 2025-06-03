import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, AsyncIterator, Optional

from aiobotocore.client import AioBaseClient
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)
T = TypeVar("T")


class AbstractS3Repository(Generic[T], ABC):
    """Generic *async repository, persisting Python objects to S3."""

    def __init__(
        self,
        s3_client: AioBaseClient,
        bucket: str,
        *,
        prefix: str = "",
        extension: str = "",
    ) -> None:
        self.s3 = s3_client
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + ("/" if prefix else "")
        self.extension = extension.lstrip(".")

    @abstractmethod
    def serialize(self, obj: T) -> bytes:
        """Convert obj to raw bytes ready for S3 upload."""

    @abstractmethod
    def deserialize(self, payload: bytes) -> T:
        """Transform raw bytes downloaded from S3 back into an object of type T."""

    def _key(self, identifier: str) -> str:
        """Convert an identifier to a full S3 key."""
        if identifier.startswith(self.prefix):  # full key provided
            return identifier
        suffix = f".{self.extension}" if self.extension and not identifier.endswith(f".{self.extension}") else ""
        return f"{self.prefix}{identifier}{suffix}"

    async def _exists(self, key: str) -> bool:
        """Check if an object with the given key exists in the S3 bucket."""
        try:
            await self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                return False
            raise  # re‑raise unexpected problems

    async def create(self, identifier: str, obj: T) -> bool:
        """Serialize obj and upload it to S3 under the given identifier."""
        key = self._key(identifier)
        data = self.serialize(obj)

        if await self._exists(key):
            return False

        await self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)
        return True

    async def load(self, identifier: str) -> Optional[T]:
        """Download and deserialize an object from S3 using the given identifier."""
        key = self._key(identifier)

        if not await self._exists(key):
            return None

        response = await self.s3.get_object(Bucket=self.bucket, Key=key)
        payload: bytes = await response["Body"].read()
        return self.deserialize(payload)

    async def delete(self, identifier: str) -> None:
        """Delete an object from S3 using the given identifier."""
        key = self._key(identifier)

        if not await self._exists(key):
            raise FileNotFoundError(f"Object '{key}' does not exist.")

        await self.s3.delete_object(Bucket=self.bucket, Key=key)

    async def list(self) -> AsyncIterator[str]:
        """List all objects in the S3 bucket matching the prefix and extension."""
        paginator = self.s3.get_paginator("list_objects_v2")

        async for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if self.extension and not key.endswith(f".{self.extension}"):
                    continue
                yield key[len(self.prefix) :].removesuffix(f".{self.extension}")
