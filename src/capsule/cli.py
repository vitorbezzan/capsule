"""Main CLI for Capsule."""

import logging
import os
import pathlib
import pickle
import typing as tp

import typer
from dotenv import load_dotenv

from capsule.capsule_mcp.server import start_mcp_server

CLI_VERSION = "202510.01"

app = typer.Typer(no_args_is_help=True)
current_dir = pathlib.Path(os.getcwd()).resolve().absolute()
logger = logging.getLogger(__name__)


def _get_environment(path: str) -> None:
    """Treats the environment variable path.

    Args:
        path: Path to the environment file.
    """
    environment_path = pathlib.Path(path).resolve().absolute()
    logger.info(f"Current working dir is {current_dir}")

    if environment_path.is_file():
        load_dotenv(path)
    else:
        raise typer.BadParameter(f"environment file not found: {path}")


def _version(value: bool) -> None:
    """Shows version information and exits.

    Args:
        value: Boolean value to show version information.
    """
    if value:
        typer.echo(f"CLI tool version v{CLI_VERSION}")
        raise typer.Exit


@app.command()
def startmcp(
    capsule_path: str = typer.Argument(..., help="Path to the capsule file."),
    model_name: str = typer.Option("instance", help="Name of the model instance."),
) -> None:
    """Starts MCP server with the given capsule.

    Args:
        capsule_path: Path to the capsule file.
        model_name: Name to be used internally for the MCP instance, and to be
            appended to all tools and actions.
    """
    path = pathlib.Path(capsule_path).resolve().absolute()

    if not path.is_file():
        raise typer.BadParameter(f"Capsule file not found {capsule_path}.")

    with open(path, "rb") as capsule_file:
        capsule = pickle.load(capsule_file)

    start_mcp_server(model_name, capsule)


@app.callback()
def main(
    environment: tp.Optional[str] = typer.Option(
        ".env",
        "--environment",
        "-e",
        help="Specific path to environment file.",
        callback=_get_environment,
        is_eager=True,
    ),
    version: tp.Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        callback=_version,
        is_eager=True,
    ),
) -> None:
    """Main entry for cli tool."""
    return


def start_cli() -> None:
    """Starts the CLI."""
    app()


if __name__ == "__main__":
    start_cli()
