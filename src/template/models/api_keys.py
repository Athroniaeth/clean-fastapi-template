import secrets

from passlib.hash import bcrypt
from sqlalchemy import Column, String, Boolean, DateTime, func, Integer

from template.database import Base

DEFAULT_ROUNDS = 12


def generate_password_hash(raw_password: str) -> str:
    """Generate a password hash using bcrypt."""
    return bcrypt.using(rounds=DEFAULT_ROUNDS).hash(raw_password)


def check_password_hash(hashed_password: str, raw_password: str) -> bool:
    """Check if the provided password matches the hashed password."""
    return bcrypt.verify(raw_password, hashed_password)


def generate_raw_key(length: int = 32) -> str:
    """Generate a random API key."""
    return secrets.token_urlsafe(length)


class ApiKeyModel(Base):
    __tablename__ = "api_keys"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    name = Column(
        String(64),
        nullable=False,
    )
    description = Column(
        String(255),
        nullable=True,
    )
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    hashed_key = Column(
        "hashed_key",
        String(128),
        nullable=False,
        unique=True,
    )

    def __init__(
        self,
        name: str,
        description: str = None,
        is_active: bool = True,
    ):
        super().__init__()
        self.name = name
        self.description = description
        self.is_active = is_active

        # Generate a random API key and hash it
        raw_key = generate_raw_key()
        self.hashed_key = generate_password_hash(raw_key)

        # Create a temporary attribute to store the raw key
        self._plain_key = raw_key

    def check_key(self, raw_key: str) -> bool:
        """Check if the provided key matches the hashed key."""
        return check_password_hash(self.hashed_key, raw_key)

    @property
    def plain_key(self) -> str:
        """
        Give the generated plain key:

        Notes:
         - only available just after instance creation.
         - avoid storing it elsewhere in the database.
        """
        # self._plain_key can be no instantiated
        plain_key = self._plain_key

        if plain_key is None:
            raise ValueError("The plain key is not available anymore.")

        # Remove the plain key after use to avoid storing it elsewhere
        del self._plain_key
        return plain_key
