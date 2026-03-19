import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import mcp.types as types
from mcp.server.fastmcp import FastMCP

from babash.client.modes import KTS
from babash.client.tool_prompts import TOOL_PROMPTS

from ...types_ import (
    Initialize,
)
from ..bash_state import CONFIG, BashState, get_tmpdir
from ..tools import (
    Context,
    default_enc,
    get_tool_output,
    parse_tool_by_name,
    which_tool_name,
)

# Log only time stamp
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger("babash")


class Console:
    def print(self, msg: str, *args: Any, **kwargs: Any) -> None:
        logger.info(msg)

    def log(self, msg: str, *args: Any, **kwargs: Any) -> None:
        logger.info(msg)


@dataclass
class AppState:
    bash_state: BashState
    custom_instructions: str | None
    console: Console


# Module-level state set by lifespan, accessed by handlers
_app_state: AppState | None = None

# Shell path set before server starts
_shell_path: str = ""


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppState]:
    """Manage BashState lifecycle — replaces global variables."""
    global _app_state
    CONFIG.update(3, 55, 5)

    custom_instructions = os.getenv("BABASH_SERVER_INSTRUCTIONS")
    console = Console()

    tmp_dir = get_tmpdir()
    starting_dir = os.path.join(tmp_dir, "claude_playground")

    with BashState(
        console,
        starting_dir,
        None,
        None,
        None,
        None,
        True,
        None,
        None,
        _shell_path or None,
    ) as bash_state:
        version = str(metadata.version("babash"))
        console.log("babash version: " + version)
        state = AppState(
            bash_state=bash_state,
            custom_instructions=custom_instructions,
            console=console,
        )
        _app_state = state
        try:
            yield state
        finally:
            _app_state = None


mcp = FastMCP("babash", lifespan=app_lifespan)

# Use the underlying low-level server for handlers with custom schemas
_server = mcp._mcp_server


def _get_app_state() -> AppState:
    assert _app_state is not None, "Server not initialized"
    return _app_state


PROMPTS = {
    "KnowledgeTransfer": (
        types.Prompt(
            name="KnowledgeTransfer",
            description="Prompt for invoking ContextSave tool in order to do a comprehensive knowledge transfer of a coding task. Prompts to save detailed error log and instructions.",
        ),
        KTS,
    )
}


@_server.list_resources()  # type: ignore
async def handle_list_resources() -> list[types.Resource]:
    return []


@_server.list_prompts()  # type: ignore
async def handle_list_prompts() -> list[types.Prompt]:
    return [x[0] for x in PROMPTS.values()]


@_server.get_prompt()  # type: ignore
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    app = _get_app_state()
    messages = [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text", text=PROMPTS[name][1][app.bash_state.mode]
            ),
        )
    ]
    return types.GetPromptResult(messages=messages)


@_server.list_tools()  # type: ignore
async def handle_list_tools() -> list[types.Tool]:
    """List available tools with custom schemas from TOOL_PROMPTS."""
    return TOOL_PROMPTS


@_server.call_tool()  # type: ignore
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if not arguments:
        raise ValueError("Missing arguments")

    app = _get_app_state()
    bash_state = app.bash_state

    tool_type = which_tool_name(name)
    tool_call = parse_tool_by_name(name, arguments)

    try:
        output_or_dones, _ = get_tool_output(
            Context(bash_state, bash_state.console),
            tool_call,
            default_enc,
            0.0,
            lambda x, y: ("", 0),
            24000,  # coding_max_tokens
            8000,  # noncoding_max_tokens
        )

    except Exception as e:
        output_or_dones = [f"GOT EXCEPTION while calling tool. Error: {e}"]

    content: list[types.TextContent | types.ImageContent | types.EmbeddedResource] = []
    for output_or_done in output_or_dones:
        if isinstance(output_or_done, str):
            if issubclass(tool_type, Initialize):
                original_message = """
- Additional important note: as soon as you encounter "The user has chosen to disallow the tool call.", immediately stop doing everything and ask user for the reason.

Initialize call done.
    """
                if app.custom_instructions:
                    output_or_done += f"\n{app.custom_instructions}\n{original_message}"
                else:
                    output_or_done += original_message

            content.append(types.TextContent(type="text", text=output_or_done))
        else:
            content.append(
                types.ImageContent(
                    type="image",
                    data=output_or_done.data,
                    mimeType=output_or_done.media_type,
                )
            )

    return content


async def main(shell_path: str = "") -> None:
    global _shell_path
    _shell_path = shell_path
    await mcp.run_stdio_async()
