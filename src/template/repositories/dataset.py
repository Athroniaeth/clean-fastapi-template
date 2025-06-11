import io

import polars as pl

from template.infrastructure.s3.adapter import AbstractS3Repository
from template.infrastructure.s3.base import S3Infrastructure


class DatasetRepository(AbstractS3Repository[pl.DataFrame]):
    """Specialised repository persisting Polars DataFrame objects as Parquet."""

    def __init__(self, s3_client: S3Infrastructure) -> None:
        super().__init__(
            s3_client,
            type_object=pl.DataFrame,
            prefix="datasets/",
            extension=".parquet",
        )

    def serialize(self, obj: pl.DataFrame) -> bytes:
        buf = io.BytesIO()
        obj.write_parquet(buf)
        return buf.getvalue()

    def deserialize(self, payload: bytes) -> pl.DataFrame:
        return pl.read_parquet(io.BytesIO(payload))
