from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class APIKeySelectSchema(BaseModel):
    """Schema for selecting an API key by its identifier.

    Attributes:
        id (int): The identifier of the API key (alias “id”).
    """

    id: int = Field(...)


class APIKeyBaseSchema(BaseModel):
    """Shared properties for API key input and output operations.

    Attributes:
        name (str): Human-readable name of the API key.
        description (Optional[str]): Optional description of the API key.
        is_active (bool): Whether the API key is active.
    """

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(..., max_length=64)
    description: Optional[str] = Field(None, max_length=255)
    is_active: bool = Field(True)


class APIKeyCreateSchema(APIKeyBaseSchema):
    """Schema for creating a new API key.

    Inherits:
        APIKeyBaseSchema: name, description, and is_active fields.
    """

    ...


class APIKeyUpdateSchema(APIKeyBaseSchema):
    """Schema for full replacement (PUT) of an existing API key.

    Attributes:
        name (str): New human-readable name.
        description (Optional[str]): New description.
        is_active (bool): New activation status.
    """

    ...


class APIKeyReadResponse(APIKeySelectSchema, APIKeyBaseSchema):
    """Schema returned for API key data in responses.

    Attributes:
        id (int): The identifier of the API key.
        name (str): Name of the API key.
        description (Optional[str]): Description of the API key.
        is_active (bool): Activation status.
        created_at (datetime): Timestamp when the key was created.
    """

    created_at: datetime


class APIKeyCreateResponse(APIKeyReadResponse):
    """Response schema for API key creation.

    Extends:
        APIKeyOutputSchema: id, name, description, is_active, created_at.

    Attributes:
        plain_key (str): The raw (unhashed) API key generated for the client.
    """

    plain_key: Optional[str] = Field(default=None, exclude=True)


class APIKeyReadResponseSchema(APIKeyReadResponse):
    model_config = {
        "json_schema_extra": {
            "examples": [
                APIKeyReadResponse(
                    id=1,
                    name="Example API Key",
                    description="This is an example API key.",
                    is_active=True,
                    created_at=datetime.now(),
                ).model_dump()
            ]
        }
    }


class APIKeyCreateResponseSchema(APIKeyCreateResponse):
    model_config = {
        "json_schema_extra": {
            "examples": [
                APIKeyCreateResponse(
                    id=1,
                    name="Example API Key",
                    description="This is an example API key.",
                    is_active=True,
                    created_at=datetime.now(),
                    plain_key="example-raw-key",
                ).model_dump()
            ]
        }
    }
