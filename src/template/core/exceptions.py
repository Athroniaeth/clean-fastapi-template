from fastapi import HTTPException
from starlette.responses import JSONResponse

from template.core.constants import Request


class APIException(HTTPException):
    """Base class for all API exceptions."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)
        self.status_code = status_code
        self.detail = detail


async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )
