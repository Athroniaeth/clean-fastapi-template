"""S3 Storage have his own module for lazy importing (aioboto3 is very slow to import)"""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from typing import AsyncIterator, Optional, Union, List

import aioboto3
from botocore.exceptions import ClientError
from types_aiobotocore_s3 import S3Client

from template.infrastructure.storage.base import AbstractStorageInfra, is_endpoint_available, DEFAULT_SERVICE_NAME


@dataclass
class S3StorageInfra(AbstractStorageInfra):
    """
    Class to handle asynchronous S3 operations using aioboto3.

    Attributes:

        bucket_name (str): Name of the S3 bucket.
        endpoint_url (str): S3 endpoint URL.
        region_name (str): AWS region name.
        aws_access_key_id (str): AWS access key ID.
        aws_secret_access_key (str): AWS secret access key.
        session (aioboto3.Session): aioboto3 session for creating clients.
    """

    bucket_name: str
    endpoint_url: str
    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str

    session: aioboto3.Session = field(default_factory=aioboto3.Session)

    async def ensure_storage_exists(self) -> None:
        """Ensure the S3 bucket exists, creating it if necessary."""
        # Check if provided endpoint URL is valid and accessible
        await is_endpoint_available(self.endpoint_url)

        if not await self.exists_bucket(self.bucket_name):
            await self.create_bucket(self.bucket_name)

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[S3Client]:
        """Async context manager to yield an S3 client."""
        async with self.session.client(
            service_name=DEFAULT_SERVICE_NAME,
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            endpoint_url=self.endpoint_url,
        ) as client:
            yield client

    async def exists_bucket(self, bucket_name: str) -> bool:
        """Check if a bucket exists.

        Args:
            bucket_name: The bucket name to check.

        Returns:
            True if the bucket exists, False otherwise.
        """
        async with self._get_client() as client:
            try:
                await client.head_bucket(Bucket=bucket_name)
                return True
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return False
                raise

    async def create_bucket(self, bucket_name: str) -> bool:
        """Create a new S3 bucket.

        Args:
            bucket_name: The name of the bucket to create.

        Raises:
            ClientError: If the bucket could not be created.
        """
        async with self._get_client() as client:
            config = {"LocationConstraint": self.region_name}
            try:
                await client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration=config,
                )
                return True
            except ClientError as e:
                if e.response["Error"]["Code"] == "BucketAlreadyExists":
                    return False
                raise

    async def _object_exists(self, client: S3Client, key: str) -> bool:
        """Check if an object exists in the bucket.

        Args:
            client: An active S3 client.
            key: Object key.

        Returns:
            True if the object exists, False otherwise.
        """
        try:
            await client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    async def exists_bytes(self, key: str) -> bool:
        """Check if an object exists by key.

        Args:
            key: Object key.

        Returns:
            True if the object exists, False otherwise.
        """
        async with self._get_client() as client:
            return await self._object_exists(client, key)

    async def get_bytes(self, key: str) -> Optional[bytes]:
        """Retrieve an object as bytes.

        Args:
            key: Object key in the bucket.

        Returns:
            The object content in bytes, or None if not found.
        """
        async with self._get_client() as client:
            try:
                response = await client.get_object(Bucket=self.bucket_name, Key=key)
                return await response["Body"].read()
            except ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                    return None
                raise

    async def create_bytes(self, key: str, data: Union[bytes, BytesIO]) -> bool:
        """Create a new object if it doesn't already exist.

        Args:
            key: Target key in the bucket.
            data: The content to upload as bytes or BytesIO.

        Returns:
            True if created, False if the object already exists.
        """
        async with self._get_client() as client:
            if await self._object_exists(client, key):
                return False
            await client.put_object(Bucket=self.bucket_name, Key=key, Body=data)
            return True

    async def update_bytes(self, key: str, data: Union[bytes, BytesIO]) -> bool:
        """Update an existing object with new data.

        Args:
            key: Target key in the bucket.
            data: New content as bytes or BytesIO.

        Returns:
            True if updated, False if object does not exist.
        """
        async with self._get_client() as client:
            if not await self._object_exists(client, key):
                return False
            await client.put_object(Bucket=self.bucket_name, Key=key, Body=data)
            return True

    async def delete_bytes(self, key: str) -> bool:
        """Delete an object by key.

        Args:
            key: Object key to delete.

        Returns:
            True if deleted, False if object does not exist.
        """
        async with self._get_client() as client:
            if not await self._object_exists(client, key):
                return False
            await client.delete_object(Bucket=self.bucket_name, Key=key)
            return True

    async def list_ids(self, prefix: str = "") -> List[str]:
        """List all objects in the bucket with an optional prefix.

        Args:
            prefix: Optional prefix to filter objects.

        Returns:
            List of object keys.
        """
        keys = []

        if not prefix.strip():
            raise ValueError("Prefix cannot be empty or whitespace.")

        if not prefix.endswith("/"):
            prefix += "/"

        async with self._get_client() as client:
            paginator = client.get_paginator("list_objects_v2")

            async for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            return keys

    async def list_bytes(self, prefix: str = "") -> List[bytes]:
        """List all objects in the bucket and return their content as bytes.

        Args:
            prefix: Optional prefix to filter objects.

        Returns:
            List of object contents in bytes.
        """
        contents = []

        if not prefix.strip():
            raise ValueError("Prefix cannot be empty or whitespace.")

        if not prefix.endswith("/"):
            prefix += "/"

        async with self._get_client() as client:
            paginator = client.get_paginator("list_objects_v2")

            async for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    response = await client.get_object(
                        Bucket=self.bucket_name,
                        Key=obj["Key"],
                    )
                    content = await response["Body"].read()
                    contents.append(content)
            return contents
