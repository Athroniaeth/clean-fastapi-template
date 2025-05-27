from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class UserSelectSchema(BaseModel):
    """Schema for selecting an User by its identifier.

    Attributes:
        id (int): The identifier of the User (alias “id”).
    """

    id: int = Field(...)


class UserByNameSelectSchema(BaseModel):
    """Schema for selecting an User by its name. (more user friendly)

    Attributes:
        username (str): The human-readable name of the User.
    """

    username: str = Field(..., max_length=64)


class UserBaseSchema(UserByNameSelectSchema):
    """Shared properties for User input and output operations.

    Attributes:
        username (str): Human-readable name of the User.
    """

    username: str = Field(..., max_length=64)
    model_config = ConfigDict(from_attributes=True)


class UserCreateSchema(UserBaseSchema):
    """Schema for creating a new User."""

    raw_password: str = Field(..., min_length=8, max_length=128)


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

    created_at: datetime


class UserReadResponseSchema(UserReadResponse):
    model_config = {
        "json_schema_extra": {
            "examples": [
                UserReadResponse(
                    id=1,
                    username="example_user",
                    created_at=datetime.now(),
                ).model_dump()
            ]
        }
    }
