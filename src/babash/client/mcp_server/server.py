import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import mcp.types as types
from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel.server import request_ctx

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
    initialize,
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
    initialized: bool = False


_shell_path: str = ""


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppState]:
    """Manage BashState lifecycle — one per MCP session."""
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
        yield AppState(
            bash_state=bash_state,
            custom_instructions=custom_instructions,
            console=console,
        )


mcp = FastMCP(
    "babash",
    lifespan=app_lifespan,
    host=os.getenv("BABASH_HOST", "127.0.0.1"),
    port=int(os.getenv("BABASH_PORT", "8000")),
)

_server = mcp._mcp_server


def _get_app_state() -> AppState:
    """Get per-session AppState from MCP request context."""
    ctx = request_ctx.get()
    state = ctx.lifespan_context
    if not isinstance(state, AppState):
        raise RuntimeError("Server not initialized — call Initialize first")
    return state


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


def _translate_tool(name: str, arguments: dict[str, Any], thread_id: str) -> tuple[str, dict[str, Any]]:
    """Translate simple tool names to internal BashCommand format."""
    if name == "RunCommand":
        return "BashCommand", {
            "type": "command",
            "command": arguments["command"],
            "is_background": arguments.get("is_background", False),
            "wait_for_seconds": arguments.get("wait_for_seconds"),
            "thread_id": thread_id,
        }
    if name == "CheckStatus":
        return "BashCommand", {
            "type": "status_check",
            "status_check": True,
            "bg_command_id": arguments.get("bg_command_id"),
            "thread_id": thread_id,
        }
    if name == "SendInput":
        return "BashCommand", {
            "type": "send_text",
            "send_text": arguments["text"],
            "bg_command_id": arguments.get("bg_command_id"),
            "thread_id": thread_id,
        }
    if name == "SendKeys":
        return "BashCommand", {
            "type": "send_specials",
            "send_specials": arguments["keys"],
            "bg_command_id": arguments.get("bg_command_id"),
            "thread_id": thread_id,
        }
    # All other tools pass through with thread_id injected
    arguments.setdefault("thread_id", thread_id)
    return name, arguments


def _auto_initialize(app: AppState) -> list[str]:
    """Auto-initialize if not done yet. Returns init output as list."""
    if app.initialized:
        return []
    app.initialized = True
    init_result, _, _ = initialize(
        "first_call",
        Context(app.bash_state, app.console),
        "",
        [],
        "",
        CODING_MAX_TOKENS,
        NONCODING_MAX_TOKENS,
        "babash",
        "",
    )
    return [init_result]


@_server.call_tool()  # type: ignore
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if not arguments:
        arguments = {}

    app = _get_app_state()
    bash_state = app.bash_state
    is_init = name == "Initialize"

    # Auto-initialize on first non-Initialize tool call
    if is_init:
        app.initialized = True
    else:
        _auto_initialize(app)

    name, arguments = _translate_tool(name, arguments, bash_state.current_thread_id)

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
        if not isinstance(output_or_done, str):
            content.append(types.ImageContent(
                type="image",
                data=output_or_done.data,
                mimeType=output_or_done.media_type,
            ))
            continue

        if is_init:
            instructions = f"\n{app.custom_instructions}" if app.custom_instructions else ""
            output_or_done += f"{instructions}\nInitialize call done.\n"

        content.append(types.TextContent(type="text", text=output_or_done))

    return content


async def main(shell_path: str = "") -> None:
    global _shell_path
    _shell_path = shell_path
    await mcp.run_stdio_async()
