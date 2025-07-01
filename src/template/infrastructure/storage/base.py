from abc import abstractmethod, ABC
from io import BytesIO
from typing import Optional, Union, List, TypeVar, Generic, Type

DEFAULT_SERVICE_NAME = "s3"


async def is_endpoint_available(endpoint_url: str, timeout: float = 1.0) -> bool:
    """Check if the endpoint is reachable."""
    import httpx

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
    async def list_ids(self, prefix: str = "") -> List[str]:
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

    @abstractmethod
    async def list_bytes(self, prefix: str = "") -> List[bytes]:
        """
        List all objects with the given prefix and return their content as bytes.

        Args:
            prefix: Path prefix.

        Returns:
            List of object content in bytes.
        """
        pass


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

    async def list_all(self) -> List[T]:
        """List all objects in the repository and return domain class"""
        list_bytes = await self.infra_file.list_bytes(prefix=self.prefix)
        return [self.deserialize(payload) for payload in list_bytes]

    async def list_ids(self) -> List[str]:
        """List all identifiers in the repository."""
        list_ids = await self.infra_file.list_ids(prefix=self.prefix)

        # Remove the prefix and extension from the keys
        slice_pre = len(self.prefix)
        slice_ext = -len(self.extension)
        list_ids = [key[slice_pre:slice_ext] for key in list_ids]

        return list_ids
