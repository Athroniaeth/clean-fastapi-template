import math
import time
from typing import Annotated, Sequence

from fastapi import (
    APIRouter,
    Depends,
    status,
)
from fastapi.params import Query
from starlette.responses import StreamingResponse

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
        result = await service.generate(
            blob=ml.blob,
            top_p=inference.top_p,
            top_k=inference.top_k,
            prompt=inference.prompt,
            max_length=inference.max_length,
            temperature=inference.temperature,
        )
        list_result.append(f"{inference.prompt}{result}")

    time_end = time.perf_counter()
    time_elapsed = round(time_end - time_start, 4)
    avg_time = round(time_elapsed / inference.n, 4)

    # Calculate "Number of Results Per Second" (NRPS)
    nrps = math.trunc(inference.n / time_elapsed)

    nbr_output_tokens = sum(len(result) for result in list_result)
    nbr_output_tokens = nbr_output_tokens - len(inference.prompt) * inference.n

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


@models_router.post(
    "/{id_}/stream",
    summary="Token-by-token generation stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Flow of generated tokens using Server-Sent Events (SSE).",
            "content": {
                "text/event-stream": {
                    "examples": {
                        "exemple": {
                            "summary": "Exemple de flux",
                            "value": ("data: B\n\ndata: o\n\ndata: i\n\nevent: end\ndata: Stream finished\n\n"),
                        }
                    },
                    # Specify the media type for SSE
                    "schema": {"type": "string", "format": "binary"},
                }
            },
        }
    },
)
async def stream_tokens(
    id_: Annotated[str, SelectModel],
    inference: Annotated[InputInference, Query],
    service: Annotated[MLService, Depends(get_service_ml)],
):
    """Stream tokens from a model using Server-Sent Events (SSE)."""
    ml = await service.get(id_)
    blob = ml.blob  # alias plus court

    # Transform async generator for SSE
    async def sse_event_generator():
        async for token_id in service.stream(
            blob=blob,
            prompt=inference.prompt,
            top_k=inference.top_k,
            top_p=inference.top_p,
            max_len=inference.max_length,
            temperature=inference.temperature,
        ):
            # In SSE format, each message must be prefixed
            # with "data: " and end with a double newline
            token_text = blob.tokenizer.decode([token_id])
            yield f"data: {token_text[0]}\n\n"

        # Indicate the end of the stream
        yield "event: end\ndata: Stream finished\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
    )
