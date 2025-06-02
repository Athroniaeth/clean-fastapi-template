from __future__ import annotations

import asyncio
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional, Type, TypeVar, Generic

from aiobotocore.client import AioBaseClient
from botocore.exceptions import BotoCoreError, ClientError

from template.database import get_s3_client
from template.settings import get_settings

T = TypeVar("T")

logger = logging.getLogger(__name__)


class S3PickleRepository(Generic[T]):
    """Repository in charge of persisting pickled objects to S3.

    This version expects the client to be injected (e.g. via FastAPI lifespan or DI).
    """

    def __init__(
        self,
        s3_client: AioBaseClient,
        type_object: Type[T],
        bucket: str,
    ) -> None:
        """
        Parameters
        ----------
        s3_client : AioBaseClient
            An existing S3 client from aioboto3 or aiobotocore.

        type_object : Type[T]
            Expected Python class for deserialization.
        """
        self.s3 = s3_client
        self.bucket = bucket
        self.type_object = type_object

    async def save(self, key: str, obj: T) -> bool:
        """Serialize and upload an object to S3."""
        try:
            payload = pickle.dumps(obj, protocol=-1)
            await self.s3.put_object(Bucket=self.bucket, Key=key, Body=payload)
            return True
        except (BotoCoreError, ClientError, pickle.PicklingError) as exc:
            logger.exception("Failed to save %s: %s", key, exc)
            return False

    async def load(self, key: str) -> Optional[T]:
        """Download and deserialize an object from S3."""
        try:
            response = await self.s3.get_object(Bucket=self.bucket, Key=key)
            data: bytes = await response["Body"].read()
            obj = pickle.loads(data)
            if not isinstance(obj, self.type_object):
                logger.error(
                    "Loaded object type %s does not match expected %s",
                    type(obj),
                    self.type_object,
                )
                return None
            return obj
        except (BotoCoreError, ClientError, pickle.UnpicklingError) as exc:
            logger.exception("Failed to load %s: %s", key, exc)
            return None

    async def delete(self, key: str) -> bool:
        """Delete an object from S3."""
        try:
            await self.s3.delete_object(Bucket=self.bucket, Key=key)
            return True
        except (BotoCoreError, ClientError) as exc:
            logger.exception("Failed to delete %s: %s", key, exc)
            return False


@dataclass
class Point:
    x: int
    y: int


async def main():
    os.environ["ENVIRONMENT"] = "development"

    async with get_s3_client() as client:
        # Ensure the bucket exists
        settings = get_settings()
        bucket = settings.s3_bucket
        region = settings.s3_region
        try:
            await client.head_bucket(Bucket=bucket)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print(f"Bucket '{bucket}' does not exist, creating it.")
                await client.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
            else:
                print("Failed to access bucket %s: %s", bucket, e)
                return

        repo = S3PickleRepository(client, type_object=Point, bucket="test-bucket")
        point = Point(x=10, y=20)
        await repo.save(key="point", obj=point)
        await repo.load(key="point")


if __name__ == "__main__":
    asyncio.run(main())
