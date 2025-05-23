from pathlib import Path
from typing import Tuple

type Version = Tuple[int, int, int]


def lint():
    """Run formatter and linter (ruff) on the codebase."""
    import subprocess

    subprocess.run("uv run ruff format .", shell=True)
    subprocess.run("uv run ruff check --fix .", shell=True)


def cli():
    """Run the CLI of the application."""
    from template.__main__ import main

    main()


# Global variables of the project
PROJECT_PATH = Path(__file__).parents[2]
DATA_PATH = PROJECT_PATH / "data"
