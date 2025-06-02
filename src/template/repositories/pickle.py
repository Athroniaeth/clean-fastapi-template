from __future__ import annotations

import logging
import pickle
from typing import Any, Optional, Type, TypeVar, Generic

import aioboto3
from botocore.exceptions import BotoCoreError, ClientError

T = TypeVar("T")

logger = logging.getLogger(__name__)


class S3PickleRepository(Generic[T]):
    """Repository in charge of persisting pickled objects to S3.

    All methods swallow internal exceptions and simply return ``False`` or
    ``None`` to signal failure, keeping the service / API layer clean.

    Attributes
    ----------
    bucket : str
        Name of the S3 bucket used for storage.
    type_object : Type[T]
        Expected type of the objects stored in this repository.
    """

    bucket: str
    type_object: Type[T]

    def __init__(
        self,
        bucket: str,
        type_object: Type[T],
        region_name: str | None = None,
        *,
        endpoint_url: str | None = None,
    ) -> None:
        self.bucket = bucket
        self.type_object = type_object
        self._session = aioboto3.Session()
        self._client_kwargs: dict[str, Any] = {}

        if region_name:
            self._client_kwargs["region_name"] = region_name
        if endpoint_url:
            # Handy for local-stack or MinIO
            self._client_kwargs["endpoint_url"] = endpoint_url

    async def save(self, key: str, obj: T) -> bool:
        """Serialize *obj* with ``pickle`` and upload it to ``bucket/key``.

        Parameters
        ----------
        key : str
            Object key in the S3 bucket.
        obj : T
            Picklable Python object.

        Returns
        -------
        bool
            ``True`` if the upload succeeded, otherwise ``False``.
        """
        try:
            payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            async with self._session.client("s3", **self._client_kwargs) as s3:
                await s3.put_object(Bucket=self.bucket, Key=key, Body=payload)
            return True
        except (BotoCoreError, ClientError, pickle.PicklingError) as exc:
            logger.exception("Failed to save %s: %s", key, exc)
            return False

    async def load(self, key: str, model: Type[T]) -> Optional[T]:
        """Download and unpickle the object at ``bucket/key``.

        The caller passes the expected *model* class so the type checker
        knows what is returned.

        Parameters
        ----------
        key : str
            Object key in the S3 bucket.
        model : Type[T]
            Expected Python class (used for typing only).

        Returns
        -------
        Optional[T]
            The unpickled instance, or ``None`` on failure / not found.
        """
        try:
            async with self._session.client("s3", **self._client_kwargs) as s3:
                response = await s3.get_object(Bucket=self.bucket, Key=key)
                data: bytes = await response["Body"].read()
            obj = pickle.loads(data)
            # Defensive: check that the object is of the expected type.
            if not isinstance(obj, model):
                logger.error(
                    "Loaded object type %s does not match expected %s",
                    type(obj),
                    model,
                )
                return None
            return obj
        except (BotoCoreError, ClientError, pickle.UnpicklingError) as exc:
            logger.exception("Failed to load %s: %s", key, exc)
            return None

    async def delete(self, key: str) -> bool:
        """Remove ``bucket/key`` from S3.

        Returns
        -------
        bool
            ``True`` if the deletion succeeded (including the *not-found*
            case, which S3 treats as success), otherwise ``False``.
        """
        try:
            async with self._session.client("s3", **self._client_kwargs) as s3:
                await s3.delete_object(Bucket=self.bucket, Key=key)
            return True
        except (BotoCoreError, ClientError) as exc:
            logger.exception("Failed to delete %s: %s", key, exc)
            return False
