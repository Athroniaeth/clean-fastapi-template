import tomllib
from pathlib import Path
from typing import Tuple

type Version = Tuple[int, int, int]


def lint():
    """Run formatter and linter (ruff) on the codebase."""
    import subprocess

    subprocess.run("uv run ruff check --fix .", shell=True)
    subprocess.run("uv run ruff format .", shell=True)
    subprocess.run("ty check src", shell=True)
    subprocess.run("vulture src --min-confidence 80", shell=True)


def cli():
    """Run the CLI of the application."""
    from template.__main__ import main

    main()


# Global variables of the project
PROJECT_PATH = Path(__file__).parents[2]
DATA_PATH = PROJECT_PATH / "data"
RAW_DATA_PATH = DATA_PATH / "raw_data"


def get_version() -> str:
    """Get the version of the application."""
    # Get the version of the pyproject.toml file
    pyproject_path = PROJECT_PATH / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")
    dict_content = tomllib.loads(content)
    return dict_content["project"]["version"]
