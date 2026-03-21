import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Literal

from mcp.server.fastmcp import Context as McpContext
from mcp.server.fastmcp import FastMCP

from babash.client.modes import KTS

from ...types_ import (
    BashCommand,
    ContextSave,
    FileWriteOrEdit,
    Initialize,
    ReadFiles,
)
from ..bash_state import CONFIG, BashState, execute_bash, get_tmpdir
from ..tools import (
    Context,
    _handle_context_save,
    _handle_initialize,
    default_enc,
    file_writing,
    initialize,
    read_files,
    read_image_from_shell,
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
    instructions="""babash is a shell and coding agent MCP server.

You have a persistent interactive terminal. Use RunCommand to execute shell commands.
The shell is stateful — cd, environment variables, and running processes persist between calls.

Key tools:
- RunCommand: execute a shell command (set is_background=true for long-running ones)
- CheckStatus: check if a command is still running, get latest output
- SendInput: send text to a running interactive program (e.g. password prompts)
- SendKeys: send special keys like Ctrl-c, Enter, arrow keys
- ReadFiles: read file contents (supports line ranges like file.py:10-20)
- FileWriteOrEdit: write or edit files using search/replace blocks
- Initialize: (optional) set workspace path, mode, or resume a task

Do not use echo/cat to read or write files — use ReadFiles and FileWriteOrEdit.
Only one foreground command runs at a time. Use CheckStatus before running a new one.
Use is_background=true for commands that run for a long time (servers, builds).
""",
    lifespan=app_lifespan,
    host=os.getenv("BABASH_HOST", "127.0.0.1"),
    port=int(os.getenv("BABASH_PORT", "8000")),
)


def _get_app(ctx: McpContext) -> AppState:  # type: ignore[type-arg]
    """Get per-session AppState from MCP context."""
    state = ctx.request_context.lifespan_context
    if not isinstance(state, AppState):
        raise RuntimeError("Server not initialized")
    return state


def _ensure_init(app: AppState) -> None:
    """Auto-initialize on first tool call."""
    if app.initialized:
        return
    app.initialized = True
    initialize(
        "first_call",
        Context(app.bash_state, app.console),
        "", [], "", CODING_MAX_TOKENS, NONCODING_MAX_TOKENS, "babash", "",
    )


def _ctx(app: AppState) -> Context:
    return Context(app.bash_state, app.console)


def _tid(app: AppState) -> str:
    return app.bash_state.current_thread_id


# --- Health ---

@mcp.custom_route("/health", methods=["GET"])  # type: ignore[untyped-decorator]
async def health_check(request: Any) -> Any:
    """Liveness/readiness probe."""
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok", "server": "babash"})


# --- Prompts ---

@mcp.prompt(name="KnowledgeTransfer", description="Save task context for knowledge transfer or resumption.")
async def knowledge_transfer(ctx: McpContext) -> str:  # type: ignore[type-arg]
    app = _get_app(ctx)
    return KTS[app.bash_state.mode]


# --- Tools ---

@mcp.tool(description="""Initialize the shell environment. Optional — auto-initializes on first tool call.
Set workspace path, execution mode, or resume a previous task.""")
async def babash_initialize(
    ctx: McpContext,  # type: ignore[type-arg]
    type: Literal["first_call", "user_asked_mode_change", "reset_shell", "user_asked_change_workspace"] = "first_call",
    any_workspace_path: str = "",
    initial_files_to_read: list[str] | None = None,
    task_id_to_resume: str = "",
    mode_name: Literal["babash", "architect", "code_writer"] = "babash",
) -> str:
    app = _get_app(ctx)
    app.initialized = True

    init_arg = Initialize(
        type=type,
        any_workspace_path=any_workspace_path,
        initial_files_to_read=initial_files_to_read or [],
        task_id_to_resume=task_id_to_resume,
        mode_name=mode_name,
    )

    output, _, _ = _handle_initialize(
        init_arg, _ctx(app), CODING_MAX_TOKENS, NONCODING_MAX_TOKENS
    )
    result = output[0]
    instructions = f"\n{app.custom_instructions}" if app.custom_instructions else ""
    return f"{result}{instructions}\nInitialize call done.\n"


@mcp.tool(description="""Execute a shell command.
Only one foreground command at a time — use CheckStatus before running another.
Set is_background=true for long-running commands (servers, builds).""")
async def run_command(
    ctx: McpContext,  # type: ignore[type-arg]
    command: str,
    is_background: bool = False,
    wait_for_seconds: float | None = None,
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)
    await ctx.info(f"$ {command}")

    bash_cmd = BashCommand.model_validate({
        "type": "command",
        "command": command,
        "is_background": is_background,
        "wait_for_seconds": wait_for_seconds,
        "thread_id": _tid(app),
    })

    output, _ = execute_bash(
        app.bash_state, default_enc, bash_cmd,
        NONCODING_MAX_TOKENS, wait_for_seconds,
    )

    # Strip echo of the command itself
    if output.startswith(command.strip()):
        output = output[len(command.strip()):]

    app.bash_state.save_state_to_disk()
    return output


@mcp.tool(description="Check if a command is still running. Returns current output and status.")
async def check_status(
    ctx: McpContext,  # type: ignore[type-arg]
    bg_command_id: str | None = None,
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)

    bash_cmd = BashCommand.model_validate({
        "type": "status_check",
        "status_check": True,
        "bg_command_id": bg_command_id,
        "thread_id": _tid(app),
    })

    output, _ = execute_bash(
        app.bash_state, default_enc, bash_cmd, NONCODING_MAX_TOKENS, None,
    )
    app.bash_state.save_state_to_disk()
    return output


@mcp.tool(description="Send text input to a running interactive program (e.g. password prompt, REPL).")
async def send_input(
    ctx: McpContext,  # type: ignore[type-arg]
    text: str,
    bg_command_id: str | None = None,
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)

    bash_cmd = BashCommand.model_validate({
        "type": "send_text",
        "send_text": text,
        "bg_command_id": bg_command_id,
        "thread_id": _tid(app),
    })

    output, _ = execute_bash(
        app.bash_state, default_enc, bash_cmd, NONCODING_MAX_TOKENS, None,
    )
    app.bash_state.save_state_to_disk()
    return output


@mcp.tool(description="Send special keys to a running program. Use Ctrl-c to interrupt, arrow keys to navigate, Enter to confirm.")
async def send_keys(
    ctx: McpContext,  # type: ignore[type-arg]
    keys: list[Literal["Enter", "Key-up", "Key-down", "Key-left", "Key-right", "Ctrl-c", "Ctrl-d"]],
    bg_command_id: str | None = None,
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)

    bash_cmd = BashCommand.model_validate({
        "type": "send_specials",
        "send_specials": keys,
        "bg_command_id": bg_command_id,
        "thread_id": _tid(app),
    })

    output, _ = execute_bash(
        app.bash_state, default_enc, bash_cmd, NONCODING_MAX_TOKENS, None,
    )
    app.bash_state.save_state_to_disk()
    return output


@mcp.tool(description="""Read content of one or more files.
Provide absolute paths (~ allowed). Supports line ranges: file.py:10-20""")
async def read_files_tool(
    ctx: McpContext,  # type: ignore[type-arg]
    file_paths: list[str],
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)

    rf = ReadFiles(file_paths=file_paths)
    result, file_ranges, _ = read_files(
        rf.file_paths, CODING_MAX_TOKENS, NONCODING_MAX_TOKENS, _ctx(app),
        rf.start_line_nums, rf.end_line_nums,
    )

    if file_ranges:
        app.bash_state.add_to_whitelist_for_overwrite(file_ranges)
    app.bash_state.save_state_to_disk()
    return result


@mcp.tool(description="Read an image file and return its contents. Provide absolute path.")
async def read_image(
    ctx: McpContext,  # type: ignore[type-arg]
    file_path: str,
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)

    image = read_image_from_shell(file_path, _ctx(app))
    return f"[Image: {image.media_type}, {len(image.data)} bytes base64]"


with open(os.path.join(os.path.dirname(__file__), "..", "diff-instructions.txt")) as _f:
    _diff_instructions = _f.read()


@mcp.tool(description="""Write or edit a file.
Set percentage_to_change: estimate what %% of existing lines will change (0-100).
If > 50: provide full file content. If <= 50: provide search/replace blocks.
Use absolute paths (~ allowed).
""" + _diff_instructions)
async def file_write_or_edit(
    ctx: McpContext,  # type: ignore[type-arg]
    file_path: str,
    percentage_to_change: int,
    text_or_search_replace_blocks: str,
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)

    fwe = FileWriteOrEdit(
        file_path=file_path,
        percentage_to_change=percentage_to_change,
        text_or_search_replace_blocks=text_or_search_replace_blocks,
        thread_id=_tid(app),
    )

    result, paths = file_writing(fwe, CODING_MAX_TOKENS, NONCODING_MAX_TOKENS, _ctx(app))

    if paths:
        app.bash_state.add_to_whitelist_for_overwrite(paths)
    app.bash_state.save_state_to_disk()
    return result


@mcp.tool(description="""Save task context and relevant files for later resumption.
Set id to a unique identifier. Set description with detailed task context in markdown.""")
async def context_save(
    ctx: McpContext,  # type: ignore[type-arg]
    id: str,
    description: str,
    relevant_file_globs: list[str],
    project_root_path: str = "",
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)

    cs = ContextSave(
        id=id,
        project_root_path=project_root_path,
        description=description,
        relevant_file_globs=relevant_file_globs,
    )
    return _handle_context_save(cs, _ctx(app))


async def main(shell_path: str = "") -> None:
    global _shell_path
    _shell_path = shell_path
    await mcp.run_stdio_async()
