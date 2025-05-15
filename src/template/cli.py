import os
import sys
from typing import Annotated, Optional

import typer
from loguru import logger
from typer import Typer

from template.constants import Level

LoggingLevel = Annotated[
    Level, typer.Option("--logging-level", "-l", help="Log level of the application.")
]

cli = Typer(
    name="template",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="CLI for the FastAPI application.",
)


def _get_workers(expected_workers: int) -> Optional[int]:
    """Get the number of workers to use."""
    max_workers = os.cpu_count()

    if expected_workers <= 0:
        logger.info(f"Uvicorn will use all possible hearts ({max_workers})")
        return max_workers

    workers = min(expected_workers, max_workers)
    logger.info(f"Uvicorn will use {workers} workers")

    if workers == 1:
        return None

    return workers


def _run(
    host: str,
    port: int,
    source: str,
    workers: int = 1,
    reload: bool = False,
):
    """Run the server using uvicorn."""
    import uvicorn

    str_host = "localhost" if host == "0.0.0.0" else host
    logger.info(f"Running server on http://{str_host}:{port}")

    # Get the amount workers available
    workers = _get_workers(workers)

    # Launch uvicorn in factory mode
    uvicorn.run(
        source,
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="warning",  # Disable uvicorn logs
        factory=True,
    )


@cli.callback()
def callback(level: LoggingLevel = Level.INFO):
    """Callback to run before any command."""
    from dotenv import load_dotenv

    # Load envvars from dotenv file
    load_dotenv()

    # Remove default logger and add a linked to stdout
    logger.remove(0)

    # Restore default logger with correct log level
    logger.add(sys.stdout, level=level)


@cli.command()
def dev(
    source: str = "template.app:create_app",
    host: str = typer.Option("localhost", envvar="HOST"),
    port: int = typer.Option(8000, envvar="PORT"),
    workers: int = typer.Option(1, envvar="WORKERS"),
):
    """Run the server in development mode."""
    _run(
        host=host,
        port=port,
        reload=True,
        source=source,
        workers=workers,
    )
