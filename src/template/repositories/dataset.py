from aiobotocore.client import AioBaseClient

from template.domain.dataset import NLPDataset
from template.infrastructure.s3 import PickleRepository


class DatasetRepository(PickleRepository[NLPDataset]):
    """Specialised repository persisting Polars DataFrame objects as Parquet."""

    def __init__(
        self,
        s3_client: AioBaseClient,
        bucket: str,
        *,
        prefix: str = "datasets/",
        raw_prefix: str = "raw/",
    ) -> None:
        super().__init__(
            s3_client,
            type_object=NLPDataset,
            bucket=bucket,
            prefix=prefix,
            extension="parquet",
        )
        self.raw_prefix = raw_prefix.rstrip("/") + "/"
