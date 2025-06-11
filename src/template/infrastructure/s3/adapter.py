import pickle
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Optional, List


from template.infrastructure.s3.base import S3Infrastructure

T = TypeVar("T")


class AbstractS3Repository(Generic[T], ABC):
    """Generic async repository, persisting Python objects to S3."""

    def __init__(
        self,
        s3_client: S3Infrastructure,
        type_object: Type[T],
        prefix: str = "",
        extension: str = "",
    ) -> None:
        # Ensure the prefix and extension are properly formatted
        list_exceptions = []

        if not (prefix := prefix.strip()):
            e = ValueError("Prefix cannot be empty or just whitespace.")
            list_exceptions.append(e)

        if not (extension := extension.strip()):
            e = ValueError("Extension cannot be empty or just whitespace.")
            list_exceptions.append(e)

        if not prefix.endswith("/"):
            e = ValueError("Prefix must end with '/'.")
            list_exceptions.append(e)

        if not extension.startswith("."):
            e = ValueError("Extension must start with a dot ('.').")
            list_exceptions.append(e)

        if list_exceptions:
            raise ExceptionGroup("Invalid S3 repository configuration", list_exceptions)

        self.s3 = s3_client
        self.type_object = type_object
        self.prefix = prefix
        self.extension = extension

    @abstractmethod
    def serialize(self, obj: T) -> bytes:
        """Convert obj to raw bytes ready for S3 upload."""

    @abstractmethod
    def deserialize(self, payload: bytes) -> T:
        """Transform raw bytes downloaded from S3 back into an object of type T."""

    def _key(self, identifier: str) -> str:
        """Convert an identifier to a full S3 key.

        Arguments:
            identifier (str): The identifier to convert.

        Returns:
            str: The full S3 key, including prefix and extension.

        Example:
            >>> class FakeRepository(AbstractS3Repository[str]):
            ...     def serialize(self, obj: str) -> bytes:
            ...         return obj.encode('utf-8')
            ...
            ...     def deserialize(self, payload: bytes) -> str:
            ...         return payload.decode('utf-8')
            ...
            >>> repo = FakeRepository(s3_client=..., type_object=str, prefix="my/prefix/", extension=".json")
            ...
            >>> repo._key("my_object")
            'my/prefix/my_object.json'

            >>> repo._key("my_object.json")
            'my/prefix/my_object.json'

            >>> repo._key("my/prefix/my_object")
            'my/prefix/my_object.json'

            >>> repo._key("my/prefix/my_object.json")
            'my/prefix/my_object.json'

        """
        if identifier.startswith(self.prefix):  # full key provided
            identifier = identifier[len(self.prefix) :]

        suffix = self.extension if not identifier.endswith(self.extension) else ""
        return f"{self.prefix}{identifier}{suffix}"

    async def exists(self, identifier: str) -> bool:
        """Check if an object with the given key exists in the S3 bucket."""
        key = self._key(identifier)
        return await self.s3.exists_bytes(key)

    async def get(self, identifier: str) -> Optional[T]:
        """Download and deserialize an object from S3 using the given identifier."""
        key = self._key(identifier)
        payload = await self.s3.get_bytes(key)

        if payload is None:
            return None

        return self.deserialize(payload)

    async def create(self, identifier: str, obj: T) -> bool:
        """Serialize obj and upload it to S3 under the given identifier."""
        key = self._key(identifier)
        data = self.serialize(obj)
        return await self.s3.create_bytes(key, data)

    async def update(self, identifier: str, obj: T) -> bool:
        """Serialize obj and update the existing object in S3 under the given identifier."""
        key = self._key(identifier)
        data = self.serialize(obj)
        return await self.s3.update_bytes(key, data)

    async def delete(self, identifier: str) -> bool:
        """Delete an object from S3 using the given identifier."""
        key = self._key(identifier)
        return await self.s3.delete_bytes(key)

    async def list(self) -> List[str]:
        """List all objects in the S3 bucket matching the prefix and extension."""
        list_ids = await self.s3.list_all(prefix=self.prefix)
        return list_ids


class PickleRepository(AbstractS3Repository, Generic[T]):
    """Repository in charge of persisting pickled objects to S3."""

    protocol: int

    def __init__(
        self,
        s3_client: S3Infrastructure,
        type_object: Type[T],
        prefix: str = "",
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> None:
        self.protocol = protocol

        super().__init__(
            s3_client,
            type_object,
            prefix=prefix,
            extension=".pickle",
        )

    def serialize(self, obj: T) -> bytes:
        """Convert obj to raw bytes ready for S3 upload."""
        return pickle.dumps(obj, protocol=self.protocol)

    def deserialize(self, payload: bytes) -> T:
        """Transform raw bytes downloaded from S3 back into an object of type T."""
        return pickle.loads(payload)
