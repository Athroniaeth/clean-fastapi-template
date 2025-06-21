from __future__ import annotations

import math
import time
from typing import Annotated, Sequence

from fastapi import (
    APIRouter,
    Depends,
    status, Path, Body,
)
from fastapi.params import Query

from template.depends import inject_s3
from template.infrastructure.storage.base import S3StorageInfra
from template.models.ml import DocumentedOutputInference, DocumentedMetadataML, DocumentedSelectModel, \
    DocumentedInputInference
from template.repositories.ml import MLRepository
from template.services.ml import MLService


async def _get_service(s3_client: Annotated[S3StorageInfra, Depends(inject_s3)]) -> MLService:
    """Return a ready-to-use service instance."""
    repository = MLRepository(s3_client)
    return MLService(repository)


models_router = APIRouter(tags=["ML"])


@models_router.get("/", status_code=status.HTTP_200_OK)
async def route_list_models(service: Annotated[MLService, Depends(_get_service)]) -> Sequence[str]:
    """List all models."""
    return await service.list()


@models_router.get("/{model_name}", status_code=status.HTTP_200_OK)
async def route_get_model(
    service: Annotated[MLService, Depends(_get_service)],
        model_name: str = "communes",
) -> DocumentedMetadataML:
    """Get a specific model by identifier."""
    ml = await service.get(model_name)
    ml.to("cpu")
    return DocumentedMetadataML(
        name=ml.__class__.__name__,
        device=str(ml.device),
    )


@models_router.post("/{model_name}/generate", status_code=status.HTTP_200_OK)
async def route_generate_model(
    inference: Annotated[DocumentedInputInference, Query],
    service: Annotated[MLService, Depends(_get_service)],
        model_name: str = "communes",
) -> DocumentedOutputInference:
    """Get a specific model by identifier."""
    list_result = []
    ml = await service.get(model_name)
    ml.to("cpu")

    time_start = time.perf_counter()
    for _ in range(inference.n):
        result = ml.generate_city_name(
            start_tokens=inference.start_tokens,
            max_length=inference.max_length,
            temperature=inference.temperature,
            top_k=inference.top_k,
            top_p=inference.top_p,
        )
        list_result.append(result)

    time_end = time.perf_counter()
    time_elapsed = round(time_end - time_start, 4)
    avg_time = round(time_elapsed / inference.n, 4)
    nrps = math.trunc(1 / time_elapsed)

    list_result = sorted(list_result)
    set_result = sorted(list(set(list_result)))

    return DocumentedOutputInference(
        name=ml.__class__.__name__,
        time_elapsed=time_elapsed,
        avg_time=avg_time,
        nrps=nrps,
        results=list_result,
        uniques=set_result,
    )
