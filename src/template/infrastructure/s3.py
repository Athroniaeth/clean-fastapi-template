import pickle
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Generic, TypeVar, Optional, List, Type, AsyncIterator

import aioboto3
from aiobotocore.client import AioBaseClient
from botocore.exceptions import ClientError
from loguru import logger


T = TypeVar("T")


class AbstractS3Repository(Generic[T], ABC):
    """Generic async repository, persisting Python objects to S3."""

    def __init__(
        self,
        s3_client: AioBaseClient,
        bucket: str,
        type_object: Type[T],
        prefix: str = "",
        extension: str = "",
    ) -> None:
        self.s3 = s3_client
        self.bucket = bucket
        self.type_object = type_object
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

    async def exists(self, key: str) -> bool:
        """Check if an object with the given key exists in the S3 bucket."""
        try:
            await self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                return False
            raise  # re‑raise unexpected problems

    async def save(self, identifier: str, obj: T) -> bool:
        """Serialize obj and upload it to S3 under the given identifier."""
        key = self._key(identifier)
        data = self.serialize(obj)
        await self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)
        return True

    async def load(self, identifier: str) -> Optional[T]:
        """Download and deserialize an object from S3 using the given identifier."""
        key = self._key(identifier)

        if not await self.exists(key):
            return None

        response = await self.s3.get_object(Bucket=self.bucket, Key=key)
        payload: bytes = await response["Body"].read()
        return self.deserialize(payload)

    async def delete(self, identifier: str) -> bool:
        """Delete an object from S3 using the given identifier."""
        key = self._key(identifier)

        if not await self.exists(key):
            return False

        await self.s3.delete_object(Bucket=self.bucket, Key=key)
        return True

    async def list(self) -> List[str]:
        """List all objects in the S3 bucket matching the prefix and extension."""
        list_identifiers = []
        paginator = self.s3.get_paginator("list_objects_v2")

        async for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if self.extension and not key.endswith(f".{self.extension}"):
                    continue
                identifier = key[len(self.prefix) :].removesuffix(f".{self.extension}")
                list_identifiers.append(identifier)

        return list_identifiers


class PickleRepository(AbstractS3Repository, Generic[T]):
    """Repository in charge of persisting pickled objects to S3."""

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


@lru_cache
@asynccontextmanager
async def create_s3(
    region: str,
    access_key_id: str,
    secret_access_key: str,
    endpoint_url: str,
    bucket: str,
) -> AsyncIterator[AioBaseClient]:
    """Get and create an S3 client with bucket creation if it does not exist."""
    s3_session = aioboto3.Session()

    async with s3_session.client(
        "s3",
        region_name=region,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        endpoint_url=f"{endpoint_url}",
    ) as s3_client:
        # Create bucket if it does not exist
        logger.debug(f"Checking if bucket {bucket} exists...")
        try:
            await s3_client.head_bucket(Bucket=bucket)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.debug(f"{bucket} does not exist, creating...")

                await s3_client.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": region},
                )
            else:
                raise

        logger.debug(f"S3 client created with endpoint: {endpoint_url}")
        yield s3_client
        logger.debug("S3 client closed.")
