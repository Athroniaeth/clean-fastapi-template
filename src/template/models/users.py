from datetime import datetime
from typing import Annotated

from passlib.hash import bcrypt  # ty: ignore[unresolved-import]
from sqlalchemy import String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from template.infrastructure.sql.base import Base

DEFAULT_ROUNDS = 12


def generate_password_hash(raw_password: str) -> str:
    """Generate a password hash using bcrypt."""
    return bcrypt.using(rounds=DEFAULT_ROUNDS).hash(raw_password)


def check_password_hash(hashed_password: str, raw_password: str) -> bool:
    """Check if the provided password matches the hashed password."""
    return bcrypt.verify(raw_password, hashed_password)


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )
    username: Mapped[Annotated[str, 64]] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )
    hashed_password: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __init__(self, username: str, raw_password: str) -> None:
        super().__init__()
        self.username = username
        self.hashed_password = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        """Validate a raw password against the hashed one.

        Args:
            raw_password (str): User input password.

        Returns:
            bool: True if match, else False.
        """
        return check_password_hash(self.hashed_password, raw_password)
