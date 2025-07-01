import asyncio
import pickle
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Generic, Type, Optional, Union, List, TypeVar

import aiofiles

from template.infrastructure.storage.base import (
    AbstractStorageInfra,
    AbstractFileRepository,
)

T = TypeVar("T")


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

    async def list_ids(self, prefix: str = "") -> List[str]:
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

    async def list_bytes(self, prefix: str = "") -> List[bytes]:
        if not prefix.strip():
            raise ValueError("Prefix cannot be empty or whitespace.")

        if not prefix.endswith("/"):
            prefix += "/"

        root = self._full_path(prefix)
        if not root.exists():
            return []

        list_bytes = []
        for path in root.rglob("*"):
            if path.is_file() and str(path).startswith(str(root)):
                async with aiofiles.open(path, mode="rb") as f:
                    content = await f.read()
                    list_bytes.append(content)

        return list_bytes


class InMemoryStorageInfra(AbstractStorageInfra):
    """
    In-memory implementation of the file infrastructure.

    This is useful for testing or temporary storage where persistence is not required.
    """

    def __init__(self) -> None:
        self.storage: dict[str, bytes] = {}

    async def ensure_storage_exists(self) -> None:
        pass  # No action needed for in-memory storage

    async def exists_bytes(self, key: str) -> bool:
        return key in self.storage

    async def get_bytes(self, key: str) -> Optional[bytes]:
        return self.storage.get(key)

    async def create_bytes(self, key: str, data: Union[bytes, BytesIO]) -> bool:
        if key in self.storage:
            return False

        self.storage[key] = data.getvalue() if isinstance(data, BytesIO) else data
        return True

    async def update_bytes(self, key: str, data: Union[bytes, BytesIO]) -> bool:
        if key not in self.storage:
            return False

        self.storage[key] = data.getvalue() if isinstance(data, BytesIO) else data
        return True

    async def delete_bytes(self, key: str) -> bool:
        if key not in self.storage:
            return False

        del self.storage[key]
        return True

    async def list_ids(self, prefix: str = "") -> List[str]:
        if not prefix.strip():
            raise ValueError("Prefix cannot be empty or whitespace.")

        return [key for key in self.storage.keys() if key.startswith(prefix)]

    async def list_bytes(self, prefix: str = "") -> List[bytes]:
        if not prefix.strip():
            raise ValueError("Prefix cannot be empty or whitespace.")

        return [self.storage[key] for key in self.storage.keys() if key.startswith(prefix)]
