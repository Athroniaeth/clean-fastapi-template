import pickle
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Optional, List


from template.infrastructure.storage.base import AbstractStorageInfra

T = TypeVar("T")


class AbstractFileRepository(Generic[T], ABC):
    """Generic async repository, persisting Python objects using StorageInfra."""

    def __init__(
        self,
        infra_file: AbstractStorageInfra,
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
            raise ExceptionGroup("Invalid StorageInfra repository configuration", list_exceptions)

        self.infra_file = infra_file
        self.type_object = type_object
        self.prefix = prefix
        self.extension = extension

    @abstractmethod
    def serialize(self, obj: T) -> bytes:
        """Convert obj to raw bytes ready for StorageInfra upload."""

    @abstractmethod
    def deserialize(self, payload: bytes) -> T:
        """Transform raw bytes downloaded from StorageInfra back into an object of type T."""

    def _get_key(self, identifier: str) -> str:
        """Convert an identifier (name) to a key (full path).

        Arguments:
            identifier (str): The identifier to convert.

        Returns:
            str: The full key, including prefix and extension.

        Example:
            >>> class FakeRepository(AbstractFileRepository[str]):
            ...     def serialize(self, obj: str) -> bytes:
            ...         return obj.encode('utf-8')
            ...
            ...     def deserialize(self, payload: bytes) -> str:
            ...         return payload.decode('utf-8')
            ...
            >>> repo = FakeRepository(s3_client=..., type_object=str, prefix="my/prefix/", extension=".json")
            ...
            >>> repo._get_key("my_object")
            'my/prefix/my_object.json'

            >>> repo._get_key("my_object.json")
            'my/prefix/my_object.json'

            >>> repo._get_key("my/prefix/my_object")
            'my/prefix/my_object.json'

            >>> repo._get_key("my/prefix/my_object.json")
            'my/prefix/my_object.json'

        """
        if identifier.startswith(self.prefix):  # full key provided
            identifier = identifier[len(self.prefix) :]

        suffix = self.extension if not identifier.endswith(self.extension) else ""
        return f"{self.prefix}{identifier}{suffix}"

    async def exists(self, identifier: str) -> bool:
        """Check if an object with the given key exists."""
        key = self._get_key(identifier)
        return await self.infra_file.exists_bytes(key)

    async def get(self, identifier: str) -> Optional[T]:
        """Download and deserialize an object using the given identifier."""
        key = self._get_key(identifier)
        payload = await self.infra_file.get_bytes(key)

        if payload is None:
            return None

        return self.deserialize(payload)

    async def create(self, identifier: str, obj: T) -> bool:
        """Serialize obj and upload it under the given identifier."""
        key = self._get_key(identifier)
        data = self.serialize(obj)
        return await self.infra_file.create_bytes(key, data)

    async def update(self, identifier: str, obj: T) -> bool:
        """Serialize obj and update the existing object under the given identifier."""
        key = self._get_key(identifier)
        data = self.serialize(obj)
        return await self.infra_file.update_bytes(key, data)

    async def delete(self, identifier: str) -> bool:
        """Delete an object from identifier."""
        key = self._get_key(identifier)
        return await self.infra_file.delete_bytes(key)

    async def list(self) -> List[str]:
        """List all identifiers in the repository."""
        list_ids = await self.infra_file.list_all(prefix=self.prefix)

        # Remove the prefix and extension from the keys
        slice_pre = len(self.prefix)
        slice_ext = -len(self.extension)
        list_ids = [key[slice_pre:slice_ext] for key in list_ids]

        return list_ids


class PickleRepository(AbstractFileRepository, Generic[T]):
    """Repository in charge of persisting pickled objects to StorageInfra."""

    protocol: int

    def __init__(
        self,
        infra_file: AbstractStorageInfra,
        type_object: Type[T],
        prefix: str = "",
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> None:
        self.protocol = protocol

        super().__init__(
            infra_file,
            type_object,
            prefix=prefix,
            extension=".pickle",
        )

    def serialize(self, obj: T) -> bytes:
        """Convert obj to raw bytes ready for StorageInfra upload."""
        return pickle.dumps(obj, protocol=self.protocol)

    def deserialize(self, payload: bytes) -> T:
        """Transform raw bytes downloaded from StorageInfra back into an object of type T."""
        return pickle.loads(payload)
