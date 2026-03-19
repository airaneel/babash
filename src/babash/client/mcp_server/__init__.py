# mypy: disable-error-code="import-untyped"
import asyncio
from importlib import metadata

import typer
from typer import Typer

from babash.client.mcp_server import server

main = Typer()


@main.command()
def app(
    version: bool = typer.Option(
        False, "--version", "-v", help="Show version and exit"
    ),
    shell: str = typer.Option(
        "", "--shell", help="Path to shell executable (defaults to $SHELL or /bin/bash)"
    ),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport: stdio or streamable-http",
    ),
) -> None:
    """babash MCP server."""
    if version:
        version_ = metadata.version("babash")
        print(f"babash version: {version_}")
        raise typer.Exit()

    server._shell_path = shell
    if transport == "streamable-http":
        server.mcp.run(transport="streamable-http")
    else:
        asyncio.run(server.main(shell))


__all__ = ["main", "server"]
