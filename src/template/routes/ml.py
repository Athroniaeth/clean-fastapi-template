from __future__ import annotations

import math
import time
from typing import Annotated, Sequence

from fastapi import (
    APIRouter,
    Depends,
    status,
)

from template.depends import inject_s3
from template.infrastructure.storage.base import S3StorageInfra
from template.repositories.ml import MLRepository
from template.services.ml import MLService


async def _get_service(s3_client: Annotated[S3StorageInfra, Depends(inject_s3)]) -> MLService:
    """Return a ready-to-use service instance."""
    repository = MLRepository(s3_client)
    return MLService(repository)


models_router = APIRouter(
    tags=["ML"],
)


@models_router.get("/", status_code=status.HTTP_200_OK)
async def route_list_models(service: Annotated[MLService, Depends(_get_service)]) -> Sequence[str]:
    """List all models."""
    return await service.list()


@models_router.get("/{model_name}", status_code=status.HTTP_200_OK)
async def route_get_model(
    service: Annotated[MLService, Depends(_get_service)],
    model_name: str,
):
    """Get a specific model by identifier."""
    ml = await service.get(model_name)
    ml.to("cpu")
    return {"name": ml.__class__.__name__, "device": str(ml.device)}


@models_router.get("/{model_name}/generate", status_code=status.HTTP_200_OK)
async def route_generate_model(
    service: Annotated[MLService, Depends(_get_service)],
    model_name: str,
    start_tokens: str = "",
    max_length: int = 20,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    n: int = 1,
):
    """Get a specific model by identifier."""
    list_result = []
    ml = await service.get(model_name)
    ml.to("cpu")

    time_start = time.perf_counter()
    for _ in range(n):
        result = ml.generate_city_name(
            start_tokens=start_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        list_result.append(result)

    time_end = time.perf_counter()
    time_elapsed = round(time_end - time_start, 4)
    avg_time = round(time_elapsed / n, 4)
    nrbd = math.trunc(1 / time_elapsed)
    nrps = math.trunc(1 / avg_time)

    list_result = sorted(list_result)
    set_result = sorted(list(set(list_result)))

    return {
        "name": ml.__class__.__name__,
        "time_elapsed": time_elapsed,
        "avg_time": avg_time,
        "nrbd": nrbd,
        "nrps": nrps,
        "result": list_result,
        "unique": set_result,
    }
