import io

import polars as pl

from template.infrastructure.storage.adapter import AbstractFileRepository
from template.infrastructure.storage.base import AbstractStorageInfra


class DatasetRepository(AbstractFileRepository[pl.DataFrame]):
    """Specialised repository persisting Polars DataFrame objects as Parquet."""

    def __init__(self, infra_storage: AbstractStorageInfra) -> None:
        super().__init__(
            infra_storage,
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
