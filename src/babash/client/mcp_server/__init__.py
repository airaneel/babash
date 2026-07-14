# mypy: disable-error-code="import-untyped"
import asyncio
from importlib import metadata

import typer
from typer import Typer

from babash.client.mcp_server import server

main = Typer()


@main.command()
def app(
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport: stdio or streamable-http",
    ),
) -> None:
    """babash MCP server.

    Configured entirely through the environment — BABASH_SHELL, BABASH_TIMEOUT,
    BABASH_HOST, … — which is read once at import (see settings.py). There is
    deliberately no flag mirroring any of it: a second way to set the same knob
    is a second place to look when it turns out wrong.
    """
    if version:
        print(f"babash version: {metadata.version('babash')}")
        raise typer.Exit()

    if transport == "streamable-http":
        server.mcp.run(transport="streamable-http")
    else:
        asyncio.run(server.main())


__all__ = ["main", "server"]
