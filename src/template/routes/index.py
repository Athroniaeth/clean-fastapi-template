from fastapi import APIRouter, Depends, Form

from template.core.state import Request
from template.routes.api_keys import keys_router
from template.schemas.users import UserReadResponse, UserCreateSchema
from template.services.auth import get_current_user, get_auth_service, AuthService

index_router = APIRouter(tags=["Utils"])
api_router = APIRouter(tags=["API"], prefix="/api/v1")
api_router.include_router(keys_router)


@index_router.get("/")
async def root(request: Request):
    """Root endpoint of the application."""
    return {
        "name": request.state.title,
        "version": request.state.version,
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@index_router.get("/health")
async def health():
    """Health endpoint of the application."""
    return {"status": "ok"}


@index_router.get("/protected")
async def protected(user: UserReadResponse = Depends(get_current_user)):
    """Protected endpoint of the application."""
    return {"status": "ok", "user": user.username}


@index_router.post("/auth/token")
async def login_for_access_token(
    username: str = Form(...),
    password: str = Form(...),
    auth_service: AuthService = Depends(get_auth_service),
):
    """Login endpoint of the application."""
    schema = UserCreateSchema(username=username, raw_password=password)

    token = await auth_service.login(
        username=schema.username,
        password=schema.raw_password,
    )
    return {"access_token": token, "token_type": "bearer"}


@index_router.post("/register")
async def register(
    schema: UserCreateSchema,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Register endpoint of the application."""
    user = await auth_service.register(schema)
    return {"status": "ok", "user": user.username}
