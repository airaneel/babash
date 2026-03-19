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

logging.basicConfig(
    level=logging.DEBUG if os.getenv("BABASH_DEBUG") else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("babash")

CODING_MAX_TOKENS = int(os.getenv("BABASH_CODING_MAX_TOKENS", "24000"))
NONCODING_MAX_TOKENS = int(os.getenv("BABASH_NONCODING_MAX_TOKENS", "8000"))


class Console:
    """Console adapter mapping print/log to Python logging levels."""

    def print(self, msg: str, *args: Any, **kwargs: Any) -> None:
        logger.info(msg)

    def log(self, msg: str, *args: Any, **kwargs: Any) -> None:
        logger.debug(msg)


@dataclass
class AppState:
    bash_state: BashState
    custom_instructions: str | None
    console: Console


# Module-level state set by lifespan, accessed by handlers.
# Needed because low-level _server handlers don't receive FastMCP context.
_app_state: AppState | None = None
_shell_path: str = ""


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppState]:
    """Manage BashState lifecycle."""
    global _app_state
    CONFIG.update(3, 55, 5)

    custom_instructions = os.getenv("BABASH_SERVER_INSTRUCTIONS")
    console = Console()

    tmp_dir = get_tmpdir()
    starting_dir = os.path.join(tmp_dir, "claude_playground")

    with BashState(
        console=console,
        working_dir=starting_dir,
        bash_command_mode=None,
        file_edit_mode=None,
        write_if_empty_mode=None,
        mode=None,
        use_screen=True,
        whitelist_for_overwrite=None,
        thread_id=None,
        shell_path=_shell_path or None,
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


mcp = FastMCP(
    "babash",
    lifespan=app_lifespan,
    host=os.getenv("BABASH_HOST", "127.0.0.1"),
    port=int(os.getenv("BABASH_PORT", "8000")),
)

_server = mcp._mcp_server


def _get_app_state() -> AppState:
    if _app_state is None:
        raise RuntimeError("Server not initialized — call Initialize first")
    return _app_state


@mcp.custom_route("/health", methods=["GET"])  # type: ignore[untyped-decorator]
async def health_check(request: Any) -> Any:
    """Liveness/readiness probe."""
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok", "server": "babash"})


PROMPTS = {
    "KnowledgeTransfer": (
        types.Prompt(
            name="KnowledgeTransfer",
            description="Save task context for knowledge transfer or resumption.",
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
            CODING_MAX_TOKENS,
            NONCODING_MAX_TOKENS,
        )
    except Exception as e:
        logger.exception("Tool call failed: %s", name)
        output_or_dones = [f"GOT EXCEPTION while calling tool. Error: {e}"]

    content: list[types.TextContent | types.ImageContent | types.EmbeddedResource] = []
    for output_or_done in output_or_dones:
        if isinstance(output_or_done, str):
            if issubclass(tool_type, Initialize):
                init_message = "\nInitialize call done.\n"
                if app.custom_instructions:
                    output_or_done += f"\n{app.custom_instructions}\n{init_message}"
                else:
                    output_or_done += init_message

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
