from contextlib import asynccontextmanager
from typing import AsyncIterator

import aioboto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from loguru import logger
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from template.settings import get_settings


url = URL.create(
    drivername="sqlite+aiosqlite",
    database="./data/db.sqlite",
)

# Create an async engine
engine = create_async_engine(
    url=url,
    future=True,
    pool_pre_ping=True,
)

# Create a session factory
async_session = sessionmaker(  # noqa
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(AsyncAttrs, DeclarativeBase):
    """Base to use for creating models."""


async def inject_db() -> AsyncIterator[AsyncSession]:
    """Dependency to inject the database session."""
    async with async_session() as session:
        yield session


@asynccontextmanager
async def get_db() -> AsyncIterator[AsyncSession]:
    """Get the database session."""
    async with async_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_database() -> None:
    """Create the database from the models."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_s3_client() -> AsyncIterator[BaseClient]:
    """Get the S3 client."""
    settings = get_settings()
    s3_session = aioboto3.Session()

    async with s3_session.client(
        "s3",
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        endpoint_url=f"{settings.s3_endpoint_url}",
    ) as s3_client:
        # Create bucket if it does not exist
        logger.debug(f"Checking if bucket {settings.s3_bucket} exists...")
        try:
            await s3_client.head_bucket(Bucket=settings.s3_bucket)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.debug(f"{settings.s3_bucket} does not exist, creating...")
                await s3_client.create_bucket(
                    Bucket=settings.s3_bucket,
                    CreateBucketConfiguration={"LocationConstraint": settings.s3_region},
                )
            else:
                raise

        logger.debug(f"S3 client created with endpoint: {settings.s3_endpoint_url}")
        yield s3_client
        logger.debug("S3 client closed.")
