import io

import polars as pl
from aiobotocore.client import AioBaseClient

from template.infrastructure.s3 import AbstractS3Repository


class DatasetRepository(AbstractS3Repository[pl.DataFrame]):
    """Specialised repository persisting Polars DataFrame objects as Parquet."""

    def __init__(
        self,
        s3_client: AioBaseClient,
        bucket: str,
        *,
        prefix: str = "datasets/",
        raw_prefix: str = "raw/",
    ) -> None:
        super().__init__(s3_client, bucket, prefix=prefix, extension="parquet")
        self.raw_prefix = raw_prefix.rstrip("/") + "/"

    def serialize(self, obj: pl.DataFrame) -> bytes:
        buf = io.BytesIO()
        obj.write_parquet(buf)
        return buf.getvalue()

    def deserialize(self, payload: bytes) -> pl.DataFrame:
        return pl.read_parquet(io.BytesIO(payload))
