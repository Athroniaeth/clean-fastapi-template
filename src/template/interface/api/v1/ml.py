import math
import time
from typing import Annotated, Sequence

from fastapi import (
    APIRouter,
    Depends,
    status,
)
from fastapi.params import Query

from template.interface.api.depends import get_service_ml

from template.interface.api.schemas.ml import (
    OutputInference,
    InputInference,
    MetadataTokenizer,
    SelectModel,
    MetadataML,
)
from template.application.ml import MLService

models_router = APIRouter(tags=["ML"])


@models_router.get("/", status_code=status.HTTP_200_OK)
async def route_list_models(service: Annotated[MLService, Depends(get_service_ml)]) -> Sequence[str]:
    """List all available models identifiers."""
    return await service.list_ids()


@models_router.get("/{id_}", status_code=status.HTTP_200_OK)
async def route_get_model(
    id_: Annotated[str, SelectModel],
    service: Annotated[MLService, Depends(get_service_ml)],
) -> MetadataML:
    """Get a specific model by identifier."""
    ml = await service.get(id_)

    return MetadataML(
        name=ml.blob.__class__.__name__,
        device=str(ml.blob.device),
        tokenizer=MetadataTokenizer(
            name=ml.blob.tokenizer.__class__.__name__,
            vocab=ml.blob.tokenizer.index_to_token,
            vocab_size=len(ml.blob.tokenizer.vocab),
        ),
    )


@models_router.post("/{id_}/generate", status_code=status.HTTP_200_OK)
async def route_generate_model(
    id_: Annotated[str, SelectModel],
    inference: Annotated[InputInference, Query],
    service: Annotated[MLService, Depends(get_service_ml)],
) -> OutputInference:
    """Generate results from a specific model using the provided inference parameters."""
    list_result = []
    time_start = time.perf_counter()
    ml = await service.get(id_)

    for _ in range(inference.n):
        result = ml.blob.generate_city_name(
            start_tokens=inference.start_tokens,
            max_length=inference.max_length,
            temperature=inference.temperature,
            top_p=inference.top_p,
        )
        list_result.append(result)

    time_end = time.perf_counter()
    time_elapsed = round(time_end - time_start, 4)
    avg_time = round(time_elapsed / inference.n, 4)

    # Calculate "Number of Results Per Second" (NRPS)
    nrps = math.trunc(inference.n / time_elapsed)

    nbr_output_tokens = sum(len(result) for result in list_result)
    nbr_output_tokens = nbr_output_tokens - len(inference.start_tokens) * inference.n

    # Calculate "Number of Tokens Per Second" (NTPS)
    ntps = math.trunc(nbr_output_tokens / time_elapsed)

    list_result = sorted(list_result)
    set_result = sorted(list(set(list_result)))

    return OutputInference(
        model_id=ml.blob.__class__.__name__,
        time_elapsed=time_elapsed,
        avg_time=avg_time,
        nrps=nrps,
        ntps=ntps,
        results=list_result,
        uniques=set_result,
    )
