from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class UserSelectSchema(BaseModel):
    """Schema for selecting an User by its identifier.

    Attributes:
        id (int): The identifier of the User (alias “id”).
    """

    id: int = Field(
        default=...,
        description="The identifier of the User.",
        examples=[1, 2, 3],
    )


class UserByNameSelectSchema(BaseModel):
    """Schema for selecting an User by its name. (more user friendly)

    Attributes:
        username (str): The human-readable name of the User.
    """

    username: str = Field(
        default=...,
        min_length=1,
        max_length=64,
        examples=["example_user", "john_doe"],
    )


class UserBaseSchema(UserByNameSelectSchema):
    """Shared properties for User input and output operations.

    Attributes:
        username (str): Human-readable name of the User.
    """

    model_config = ConfigDict(from_attributes=True)


class UserCreateSchema(UserBaseSchema):
    """Schema for creating a new User."""

    raw_password: str = Field(
        default=...,
        min_length=8,
        max_length=128,
        examples=["securepassword123"],
    )


class UserUpdateSchema(UserBaseSchema):
    """Schema for full replacement (PUT) of an existing User."""

    ...


class UserReadResponse(UserSelectSchema, UserBaseSchema):
    """Schema returned for User data in responses.

    Attributes:
        id (int): The identifier of the User.
        username (str): Name of the User.
        created_at (datetime): Timestamp when the key was created.
    """

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the User was created.",
        examples=[datetime.now().isoformat()],
    )
