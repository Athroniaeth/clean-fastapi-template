import asyncio
from abc import abstractmethod, ABC
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import AsyncIterator, Optional, Union, List

import aioboto3
import aiofiles
import httpx
from botocore.exceptions import ClientError
from types_aiobotocore_s3.client import S3Client
from types_aiobotocore_s3.literals import BucketLocationConstraintType


DEFAULT_SERVICE_NAME = "s3"


async def is_endpoint_available(endpoint_url: str, timeout: float = 1.0) -> bool:
    """Check if the endpoint is reachable."""
    try:
        httpx.get(endpoint_url, timeout=timeout)
        return True
    except KeyboardInterrupt:
        raise ValueError(
            f"Cannot reach S3 endpoint: {endpoint_url}. If you're in dev, did you forget to run 'moto_server -p 5000'?"
        )


class AbstractStorageInfra(ABC):
    """Abstract interface for file storage infrastructure."""

    @abstractmethod
    async def ensure_storage_exists(self) -> None:
        """
        Ensure the storage backend is initialized and available.
        """
        pass

    @abstractmethod
    async def exists_bytes(self, key: str) -> bool:
        """
        Check if an object exists.

        Args:
            key: Object key.

        Returns:
            True if the object exists, False otherwise.
        """
        pass

    @abstractmethod
    async def get_bytes(self, key: str) -> Optional[bytes]:
        """
        Retrieve the object content as bytes.

        Args:
            key: Object key.

        Returns:
            Content in bytes if found, else None.
        """
        pass

    @abstractmethod
    async def create_bytes(self, key: str, data: Union[bytes, BytesIO]) -> bool:
        """
        Create a new object.

        Args:
            key: Object key.
            data: Object content.

        Returns:
            True if the object was created, False if it already exists.
        """
        pass

    @abstractmethod
    async def update_bytes(self, key: str, data: Union[bytes, BytesIO]) -> bool:
        """
        Update an existing object.

        Args:
            key: Object key.
            data: New object content.

        Returns:
            True if updated, False if the object does not exist.
        """
        pass

    @abstractmethod
    async def delete_bytes(self, key: str) -> bool:
        """
        Delete an object.

        Args:
            key: Object key.

        Returns:
            True if the object was deleted, False if it does not exist.
        """
        pass

    @abstractmethod
    async def list_all(self, prefix: str = "") -> List[str]:
        """
        List all objects with the given prefix.

        Notes:
            This method return key with the prefix and the extension.

        Args:
            prefix: Path prefix.

        Returns:
            List of object keys.
        """
        pass


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
    region_name: BucketLocationConstraintType

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
            try:
                await client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={
                        "LocationConstraint": self.region_name,
                    },
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

    async def list_all(self, prefix: str = "") -> List[str]:
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


@dataclass
class LocalStorageInfra(AbstractStorageInfra):
    """
    Local filesystem-based implementation of the file infrastructure.

    All object keys are interpreted as relative paths under the base directory.

    Args:
        base_path: Base directory for file storage.
    """

    base_path: Path

    def __init__(self, base_path: Union[str, Path]) -> None:
        self.base_path = Path(base_path)
        self.base_path = self.base_path.resolve()
        self.base_path = self.base_path.absolute()

    def _full_path(self, key: str) -> Path:
        return self.base_path / key.lstrip("/")

    async def ensure_storage_exists(self) -> None:
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def exists_bytes(self, key: str) -> bool:
        return self._full_path(key).exists()

    async def get_bytes(self, key: str) -> Optional[bytes]:
        path = self._full_path(key)
        if not path.exists():
            return None
        async with aiofiles.open(path, mode="rb") as f:
            return await f.read()

    async def create_bytes(self, key: str, data: Union[bytes, BytesIO]) -> bool:
        path = self._full_path(key)
        if path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, mode="wb") as f:
            await f.write(data.getvalue() if isinstance(data, BytesIO) else data)
        return True

    async def update_bytes(self, key: str, data: Union[bytes, BytesIO]) -> bool:
        path = self._full_path(key)
        if not path.exists():
            return False
        async with aiofiles.open(path, mode="wb") as f:
            await f.write(data.getvalue() if isinstance(data, BytesIO) else data)
        return True

    async def delete_bytes(self, key: str) -> bool:
        path = self._full_path(key)
        if not path.exists():
            return False
        await asyncio.to_thread(path.unlink)
        return True

    async def list_all(self, prefix: str = "") -> List[str]:
        if not prefix.strip():
            raise ValueError("Prefix cannot be empty or whitespace.")

        if not prefix.endswith("/"):
            prefix += "/"

        root = self._full_path(prefix)
        if not root.exists():
            return []

        list_ids = [
            f"{path.relative_to(self.base_path)}"
            for path in root.rglob("*")
            if path.is_file() and str(path).startswith(str(root))
        ]

        return list_ids
