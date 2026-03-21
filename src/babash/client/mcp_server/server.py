import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Literal

from mcp.server.fastmcp import Context as McpContext, FastMCP
from mcp.server.lowlevel.server import request_ctx
from mcp.types import ToolAnnotations

from babash.client.modes import KTS

from ...types_ import (
    BashCommand,
    ContextSave,
    FileWriteOrEdit,
    Initialize,
    ReadFiles,
    WriteIfEmpty,
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
from ..tools.write_ops import write_file

logging.basicConfig(
    level=logging.DEBUG if os.getenv("BABASH_DEBUG") else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("babash")

CODING_MAX_TOKENS = int(os.getenv("BABASH_CODING_MAX_TOKENS", "32000"))
NONCODING_MAX_TOKENS = int(os.getenv("BABASH_NONCODING_MAX_TOKENS", "16000"))


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
    sessions: dict[str, BashState] | None = None

    def get_sessions(self) -> dict[str, BashState]:
        if self.sessions is None:
            self.sessions = {}
        return self.sessions

    def get_shell(self, session: str | None) -> BashState:
        """Get BashState for a named session, or main shell if None."""
        if not session or session == "main":
            return self.bash_state
        sessions = self.get_sessions()
        if session not in sessions:
            raise ValueError(
                f"Session '{session}' not found. "
                f"Available: main, {', '.join(sessions.keys()) or '(none)'}. "
                f"Create one with create_session."
            )
        return sessions[session]


_shell_path: str = ""


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppState]:
    """Manage BashState lifecycle — one per MCP session."""
    CONFIG.update(
        timeout=float(os.getenv("BABASH_TIMEOUT", "2")),
        timeout_while_output=float(os.getenv("BABASH_TIMEOUT_WHILE_OUTPUT", "15")),
        output_wait_patience=float(os.getenv("BABASH_OUTPUT_PATIENCE", "3")),
    )

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
        app = AppState(
            bash_state=bash_state,
            custom_instructions=custom_instructions,
            console=console,
        )
        try:
            yield app
        finally:
            # Graceful shutdown: clean up all named sessions
            for name, shell in app.get_sessions().items():
                try:
                    shell.cleanup()
                    console.log(f"Session '{name}' cleaned up")
                except Exception as e:
                    console.log(f"Error cleaning up session '{name}': {e}")


mcp = FastMCP(
    "babash",
    instructions="""babash is a shell and coding agent MCP server with multiple persistent terminals.

# Shell commands
- run_command(command): execute a command and get output.
- check_status(): check if the last command is still running.
- send_input(text): send text to a running interactive program (passwords, prompts).
- send_keys(keys): send special keys — "Ctrl-c" to interrupt, "Enter" to confirm, arrow keys to navigate.

# Sessions (parallel shells)
You start with a 'main' session. For parallel work, create named sessions:
- create_session(name="server") → independent shell
- run_command(command="npm start", session="server")
- check_status(session="server")
- send_keys(keys="Ctrl-c", session="server")
- destroy_session(name="server") → clean up when done

Use sessions when you need things running simultaneously:
  session "server" → npm start (keeps running)
  session "main"   → npm test (while server runs)

Use list_sessions() to see all sessions with their status and last command.

# Background commands (within a session)
For fire-and-forget commands within one session:
- run_command(command="long-build", is_background=true) → returns bg_command_id
- check_status(bg_command_id="...") → check progress

Key difference: sessions are persistent independent shells. Background commands are
one-off processes within a session.

# File operations
- read_files_tool(file_paths): read files. Supports line ranges: file.py:10-20
- create_file(file_path, content): create a new file (fails if exists)
- file_write_or_edit(file_path, percentage_to_change, text_or_search_replace_blocks): edit existing files

Do NOT use echo/cat/sed to read or write files — use the file tools instead.

# Important
- Each session runs one foreground command at a time.
- If a command is still running, check_status or send_keys(Ctrl-c) before running another.
- cd, env vars, and state persist within each session independently.
- If output is truncated, use more precise commands (grep, head, tail, awk) instead of dumping everything.
- For large files, use read_files_tool with line ranges (file.py:1-50) instead of reading the whole file.
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


# --- Resources ---

def _get_app_from_request() -> AppState:
    """Get AppState from request context (for resources which don't get ctx)."""
    ctx = request_ctx.get()
    state = ctx.lifespan_context
    if not isinstance(state, AppState):
        raise RuntimeError("Server not initialized")
    return state


@mcp.resource("babash://workspace/tree", description="Current workspace directory tree")
def workspace_tree() -> str:
    app = _get_app_from_request()
    _ensure_init(app)
    workspace = app.bash_state.workspace_root or app.bash_state.cwd
    try:
        from ..repo_ops.repo_context import get_repo_context
        tree, _ = get_repo_context(workspace)
        return f"Workspace: {workspace}\n\n{tree}"
    except Exception:
        return f"Workspace: {workspace}\n(unable to generate tree)"


@mcp.resource("babash://workspace/env", description="Shell environment and system info")
def workspace_env() -> str:
    import platform
    import shutil

    app = _get_app_from_request()
    _ensure_init(app)
    bs = app.bash_state

    # System info
    lines = [
        f"system: {platform.system()} {platform.release()}",
        f"machine: {platform.machine()}",
        f"shell_cwd: {bs.cwd}",
        f"workspace_root: {bs.workspace_root}",
        f"mode: {bs.mode}",
        f"state: {bs.state}",
    ]

    # Detect available tools
    for tool in ["git", "docker", "python3", "node", "npm", "uv", "pip", "rg", "jq", "ssh", "curl"]:
        path = shutil.which(tool)
        if path:
            lines.append(f"has_{tool}: {path}")

    return "\n".join(lines)


@mcp.resource("babash://workspace/processes", description="All sessions and running commands")
def workspace_processes() -> str:
    app = _get_app_from_request()
    _ensure_init(app)
    lines = [f"main: cwd={app.bash_state.cwd} state={app.bash_state.state} cmd={app.bash_state.last_command or '(idle)'}"]
    for name, shell in app.get_sessions().items():
        lines.append(f"{name}: cwd={shell.cwd} state={shell.state} cmd={shell.last_command or '(idle)'}")
    bg = app.bash_state.background_shells
    for cid, state in bg.items():
        lines.append(f"bg/{cid}: {state.last_command} (state={state.state})")
    return "\n".join(lines)


# --- Session Management ---

@mcp.tool(
    description="""Create a named shell session. Each session is an independent persistent shell.
Use sessions to run multiple things in parallel (e.g. a server + tests + build).
The 'main' session always exists.""",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False),
)
async def create_session(
    ctx: McpContext,  # type: ignore[type-arg]
    name: str,
    working_directory: str = "",
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)
    sessions = app.get_sessions()

    if name == "main":
        return "Error: 'main' session already exists."
    if name in sessions:
        return f"Session '{name}' already exists. Use it with session='{name}' on run_command etc."

    cwd = working_directory or app.bash_state.cwd
    new_shell = BashState(
        console=app.console,
        working_dir=cwd,
        bash_command_mode=None,
        file_edit_mode=None,
        write_if_empty_mode=None,
        mode=None,
        use_screen=True,
        whitelist_for_overwrite=None,
        thread_id=None,
        shell_path=_shell_path or None,
    )
    sessions[name] = new_shell
    return f"Session '{name}' created (cwd: {cwd}). Use session='{name}' on run_command, check_status, send_input, send_keys."


@mcp.tool(
    description="List all shell sessions and their status.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
)
async def list_sessions(
    ctx: McpContext,  # type: ignore[type-arg]
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)

    lines = [f"- main: cwd={app.bash_state.cwd} state={app.bash_state.state} (default)"]
    for name, shell in app.get_sessions().items():
        lines.append(f"- {name}: cwd={shell.cwd} state={shell.state} last_cmd={shell.last_command or '(none)'}")
    return "\n".join(lines)


@mcp.tool(
    description="Destroy a named session. Sends Ctrl-c to any running command, then cleans up the shell.",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False),
)
async def destroy_session(
    ctx: McpContext,  # type: ignore[type-arg]
    name: str,
) -> str:
    app = _get_app(ctx)
    if name == "main":
        return "Error: cannot destroy the main session."
    sessions = app.get_sessions()
    if name not in sessions:
        return f"Session '{name}' not found."
    shell = sessions.pop(name)
    try:
        shell.sendintr()
        shell.cleanup()
    except Exception:
        pass
    return f"Session '{name}' destroyed."


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

@mcp.tool(
    description="""Initialize the shell environment. Optional — auto-initializes on first tool call.
Set workspace path, execution mode, or resume a previous task.""",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
)
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


@mcp.tool(
    description="""Execute a shell command.
Only one foreground command at a time — use check_status before running another.
Set is_background=true for long-running commands (servers, builds).""",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True),
)
async def run_command(
    ctx: McpContext,  # type: ignore[type-arg]
    command: str,
    is_background: bool = False,
    wait_for_seconds: float | None = None,
    session: str | None = None,
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)
    shell = app.get_shell(session)
    # Check for dangerous commands and elicit confirmation
    dangerous_patterns = ["rm -rf", "rm -r /", "mkfs", "dd if=", "> /dev/", ":(){ :|:& };:"]
    if any(p in command for p in dangerous_patterns):
        try:
            from pydantic import BaseModel as _BM

            class Confirm(_BM):
                proceed: bool = False

            result = await ctx.elicit(
                f"⚠️ Dangerous command detected: `{command}`\nProceed?",
                Confirm,
            )
            from mcp.server.elicitation import AcceptedElicitation
            if not isinstance(result, AcceptedElicitation) or not result.data.proceed:
                return "Command cancelled by user."
        except Exception:
            pass  # Client doesn't support elicitation — proceed

    await ctx.info(f"$ {command}")
    await ctx.report_progress(0, 1, "executing...")

    # If shell is busy, tell the LLM instead of failing with a cryptic error
    if shell.state == "pending" and not is_background:
        running = shell.last_command or "unknown"
        pending_for = shell.get_pending_for()
        sname = session or "main"
        return (
            f"Cannot run command — session '{sname}' has a command still running.\n"
            f"Running: {running}\n"
            f"Running for: {pending_for}\n\n"
            f"Options:\n"
            f"1. Use `check_status(session='{sname}')` to see if it finished\n"
            f"2. Use `send_keys(keys='Ctrl-c', session='{sname}')` to interrupt it\n"
            f"3. Run this command in a different session or with is_background=true"
        )

    bash_cmd = BashCommand.model_validate({
        "type": "command",
        "command": command,
        "is_background": is_background,
        "wait_for_seconds": wait_for_seconds,
        "thread_id": shell.current_thread_id,
    })

    output, _ = execute_bash(
        shell, default_enc, bash_cmd,
        NONCODING_MAX_TOKENS, wait_for_seconds,
    )

    # Strip echo of the command itself
    if output.startswith(command.strip()):
        output = output[len(command.strip()):]

    await ctx.report_progress(1, 1, "done")
    shell.save_state_to_disk()
    return output


@mcp.tool(
    description="Check if a command is still running. Returns current output and status.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
)
async def check_status(
    ctx: McpContext,  # type: ignore[type-arg]
    bg_command_id: str | None = None,
    session: str | None = None,
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)
    shell = app.get_shell(session)

    bash_cmd = BashCommand.model_validate({
        "type": "status_check",
        "status_check": True,
        "bg_command_id": bg_command_id,
        "thread_id": shell.current_thread_id,
    })

    output, _ = execute_bash(
        shell, default_enc, bash_cmd, NONCODING_MAX_TOKENS, None,
    )
    shell.save_state_to_disk()
    return output


@mcp.tool(
    description="Send text input to a running interactive program (e.g. password prompt, REPL).",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False),
)
async def send_input(
    ctx: McpContext,  # type: ignore[type-arg]
    text: str,
    bg_command_id: str | None = None,
    session: str | None = None,
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)
    shell = app.get_shell(session)

    bash_cmd = BashCommand.model_validate({
        "type": "send_text",
        "send_text": text,
        "bg_command_id": bg_command_id,
        "thread_id": shell.current_thread_id,
    })

    output, _ = execute_bash(
        shell, default_enc, bash_cmd, NONCODING_MAX_TOKENS, None,
    )
    shell.save_state_to_disk()
    return output


@mcp.tool(
    description="Send special keys to a running program. Use Ctrl-c to interrupt, arrow keys to navigate, Enter to confirm.",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False),
)
async def send_keys(
    ctx: McpContext,  # type: ignore[type-arg]
    keys: list[str] | str = "Ctrl-c",
    bg_command_id: str | None = None,
    session: str | None = None,
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)
    shell = app.get_shell(session)

    keys_list = [keys] if isinstance(keys, str) else keys

    bash_cmd = BashCommand.model_validate({
        "type": "send_specials",
        "send_specials": keys_list,
        "bg_command_id": bg_command_id,
        "thread_id": shell.current_thread_id,
    })

    output, _ = execute_bash(
        shell, default_enc, bash_cmd, NONCODING_MAX_TOKENS, None,
    )
    shell.save_state_to_disk()
    return output


@mcp.tool(
    description="""Read content of one or more files.
Provide absolute paths (~ allowed). Supports line ranges: file.py:10-20""",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
)
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


@mcp.tool(
    description="Read an image file and return its contents. Provide absolute path.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
)
async def read_image(
    ctx: McpContext,  # type: ignore[type-arg]
    file_path: str,
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)

    image = read_image_from_shell(file_path, _ctx(app))
    return f"[Image: {image.media_type}, {len(image.data)} bytes base64]"


@mcp.tool(
    description="""Create a new file with the given content. Use absolute paths (~ allowed).
Fails if the file already exists — use file_write_or_edit to modify existing files.""",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False),
)
async def create_file(
    ctx: McpContext,  # type: ignore[type-arg]
    file_path: str,
    content: str,
) -> str:
    app = _get_app(ctx)
    _ensure_init(app)

    wf = WriteIfEmpty(file_path=file_path, file_content=content)
    result, paths = write_file(wf, True, CODING_MAX_TOKENS, NONCODING_MAX_TOKENS, _ctx(app))

    if paths:
        app.bash_state.add_to_whitelist_for_overwrite(paths)
    app.bash_state.save_state_to_disk()
    return result


with open(os.path.join(os.path.dirname(__file__), "..", "diff-instructions.txt")) as _f:
    _diff_instructions = _f.read()


@mcp.tool(
    description="""Edit an existing file using search/replace blocks.
Set percentage_to_change: estimate what %% of existing lines will change (0-100).
If > 50: provide full file content. If <= 50: provide search/replace blocks.
Use absolute paths (~ allowed).
""" + _diff_instructions,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False),
)
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


@mcp.tool(
    description="""Save task context and relevant files for later resumption.
Set id to a unique identifier. Set description with detailed task context in markdown.""",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False),
)
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
