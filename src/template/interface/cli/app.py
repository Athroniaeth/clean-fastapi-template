import os
import sys
from enum import StrEnum
from typing import Annotated, Optional

import typer
from loguru import logger

from template.interface.cli.commands.api_keys import cli_keys
from template.interface.cli.commands.dataset import cli_dataset
from template.interface.cli.commands.ml import cli_ml
from template.interface.cli.commands.tokenizer import cli_tokenizer
from template.interface.cli.commands.users import cli_users
from template.core.cli import AsyncTyper

from typing import Callable, AsyncContextManager

type Lifespan = Callable[[], AsyncContextManager[None]]


class Level(StrEnum):
    """
    Log levels used to trace application execution.

    Attributes:
        TRACE   : Very fine-grained details for deep debugging.
        DEBUG   : Debugging information for developers.
        INFO    : General operational events.
        WARNING : Unexpected behavior that doesn't stop execution.
        ERROR   : Critical issues that affect functionality.
    """

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


LoggingLevel = Annotated[
    Level,
    typer.Option(
        "--logging-level",
        "-l",
        help="Log level of the application.",
    ),
]

cli = AsyncTyper(
    name="template",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="CLI for the FastAPI application.",
)
cli.add_typer(cli_keys)
cli.add_typer(cli_dataset)
cli.add_typer(cli_tokenizer)
cli.add_typer(cli_users)
cli.add_typer(cli_ml)


def _get_workers(expected_workers: Optional[int]) -> Optional[int]:
    """Get the number of workers to use."""
    max_workers = os.cpu_count()

    if not expected_workers:
        logger.info("Uvicorn don't use workers.")
        return None

    if not expected_workers <= 0:
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
    workers: Optional[int],
    reload: bool = False,
):
    """Run the server using uvicorn.

    Args:
        host (str): Host to bind the server.
        port (int): Port to bind the server.
        source (str): Source of the FastAPI app in factory mode.
        reload (bool): Whether to enable auto-reload for development.
        workers (Optional[int]): Number of workers to use (None for no workers).
    """
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
        factory=True,  # Use factory mode to create the app
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
    source: str = typer.Option("template.interface.api.app:factory_app"),
    host: str = typer.Option("localhost", envvar="HOST", help="Host to bind the server"),
    port: int = typer.Option(8000, envvar="PORT", help="Port to bind the server"),
    reload: bool = typer.Option(False, envvar="RELOAD", help="Enable auto-reload for development"),
    workers: Optional[int] = typer.Option(None, envvar="WORKERS", help="Number of workers to use (None for no workers)"),
):
    """Run the server in development mode."""
    _run(
        host=host,
        port=port,
        reload=reload,
        source=source,
        workers=workers,
    )
