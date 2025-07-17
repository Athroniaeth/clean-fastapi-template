from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict

from rename.domain.api_keys import generate_raw_key


class APIKeySelect(BaseModel):
    """Schema for selecting an API key by its identifier.

    Attributes:
        id_ (int): The identifier of the API key (alias “id”).
    """

    id_: int = Field(
        default=...,
        description="The identifier of the API key.",
        examples=[1, 2, 3],
    )


class APIKeySchema(BaseModel):
    """Shared properties for API key input and output operations.

    Attributes:
        name (str): Human-readable name of the API key.
        description (Optional[str]): Optional description of the API key.
        is_active (bool): Whether the API key is active.
    """

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(
        default=...,
        max_length=64,
        description="Human-readable name of the API key.",
        examples=["dev", "production"],
    )
    description: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Optional description of the API key.",
    )
    is_active: bool = Field(
        default=True,
        description="Indicates whether the API key is currently active.",
        examples=[True, False],
    )


class APIKeyCreate(APIKeySchema):
    """Schema for full replacement (PUT) of an existing API key.

    Attributes:
        name (str): New human-readable name.
        description (Optional[str]): New description.
        is_active (bool): New activation status.
    """

    ...


class APIKeyUpdate(APIKeySchema, APIKeySelect):
    """Schema for full replacement (PUT) of an existing API key.

    Attributes:
        id_ (int): The identifier of the API key.
        name (str): New human-readable name.
        description (Optional[str]): New description.
        is_active (bool): New activation status.
    """

    ...


class APIKeyRead(APIKeySelect, APIKeySchema):
    """Schema returned for API key data in responses.

    Attributes:
        id (int): The identifier of the API key.
        name (str): Name of the API key.
        description (Optional[str]): Description of the API key.
        is_active (bool): Activation status.
        created_at (datetime): Timestamp when the key was created.
    """

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the API key was created.",
        examples=[datetime.now().isoformat()],
    )


class APIKeyCreateResponse(APIKeyRead):
    """Response schema for API key creation.

    Extends:
        APIKeyOutputSchema: id, name, description, is_active, created_at.

    Attributes:
        plain_key (str): The raw (unhashed) API key generated for the client.
    """

    plain_key: Optional[str] = Field(
        default=None,
        max_length=64,
        description=(
            "The raw (unhashed) API key generated for the client. "
            "This is a one-time value that should be stored securely by the client. "
            "It will not be returned again after creation."
        ),
        examples=[generate_raw_key()],
    )
