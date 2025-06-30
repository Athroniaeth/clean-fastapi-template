import logging

from dotenv import load_dotenv
from loguru import logger


def main():
    """Entry point for the application."""
    from template.interface.cli.app import cli

    # Disable useless warnings
    # https://github.com/pyca/bcrypt/issues/684#issuecomment-1858400267
    logging.getLogger("passlib").setLevel(logging.ERROR)

    # Loading environment variables from .env file
    load_dotenv()

    try:
        cli()
    except KeyboardInterrupt:
        logger.info("Interrupt signal received. Exiting...")
        raise SystemExit(0)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e
    else:
        logger.info("Finished program execution.")
        raise SystemExit(0)


if __name__ == "__main__":
    app = main()
