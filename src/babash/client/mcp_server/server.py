"""babash MCP server — assembly only.

The FastMCP instance lives in instance.py; tools and resources register
themselves against it when their modules are imported. This module's whole job
is to make sure that happens, and then to run the thing.
"""

from . import resources, tools  # noqa: F401  — imported for registration
from .instance import mcp

__all__ = ["main", "mcp"]


async def main() -> None:
    await mcp.run_stdio_async()
