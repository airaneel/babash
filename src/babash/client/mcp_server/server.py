"""babash MCP server — wiring only. All logic lives in helpers/state/tools."""

import logging
import os
import shlex
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import metadata
from typing import Any, Literal

from mcp.server.fastmcp import Context as McpContext, FastMCP
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
from .helpers import CODING_MAX_TOKENS, NONCODING_MAX_TOKENS, detect_errors, get_incremental, record_command
from .state import AppState, Console

logging.basicConfig(
    level=logging.DEBUG if os.getenv("BABASH_DEBUG") else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_shell_path: str = os.getenv("BABASH_SHELL", "")


# --- Lifespan ---

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppState]:
    CONFIG.update(
        timeout=float(os.getenv("BABASH_TIMEOUT", "2")),
        timeout_while_output=float(os.getenv("BABASH_TIMEOUT_WHILE_OUTPUT", "15")),
        output_wait_patience=float(os.getenv("BABASH_OUTPUT_PATIENCE", "3")),
    )
    console = Console()
    tmp_dir = get_tmpdir()

    with BashState(
        console=console, working_dir=os.path.join(tmp_dir, "claude_playground"),
        bash_command_mode=None, file_edit_mode=None, write_if_empty_mode=None,
        mode=None, use_screen=True, whitelist_for_overwrite=None,
        thread_id=None, shell_path=_shell_path or None,
    ) as bash_state:
        console.log("babash version: " + str(metadata.version("babash")))
        app = AppState(
            bash_state=bash_state,
            custom_instructions=os.getenv("BABASH_SERVER_INSTRUCTIONS"),
            console=console,
        )
        try:
            yield app
        finally:
            for name, shell in app.get_sessions().items():
                try:
                    shell.cleanup()
                except Exception:
                    pass


# --- FastMCP instance ---

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


def get_app(ctx: McpContext) -> AppState:  # type: ignore[type-arg]
    state = ctx.request_context.lifespan_context
    if not isinstance(state, AppState):
        raise RuntimeError("Server not initialized")
    return state


def get_app_from_request() -> AppState:
    from mcp.server.lowlevel.server import request_ctx
    state = request_ctx.get().lifespan_context
    if not isinstance(state, AppState):
        raise RuntimeError("Server not initialized")
    return state


def ensure_init(app: AppState) -> None:
    if app.initialized:
        return
    app.initialized = True
    initialize("first_call", Context(app.bash_state, app.console),
               "", [], "", CODING_MAX_TOKENS, NONCODING_MAX_TOKENS, "babash", "")


def make_context(app: AppState) -> Context:
    return Context(app.bash_state, app.console)


# --- Resources ---

@mcp.resource("babash://workspace/tree", description="Current workspace directory tree")
def workspace_tree() -> str:
    app = get_app_from_request()
    ensure_init(app)
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
    app = get_app_from_request()
    ensure_init(app)
    bs = app.bash_state
    lines = [
        f"system: {platform.system()} {platform.release()}",
        f"machine: {platform.machine()}",
        f"shell_cwd: {bs.cwd}",
        f"workspace_root: {bs.workspace_root}",
        f"mode: {bs.mode}",
        f"state: {bs.state}",
    ]
    for tool in ["git", "docker", "python3", "node", "npm", "uv", "pip", "rg", "jq", "ssh", "curl"]:
        path = shutil.which(tool)
        if path:
            lines.append(f"has_{tool}: {path}")
    return "\n".join(lines)


@mcp.resource("babash://workspace/processes", description="All sessions and running commands")
def workspace_processes() -> str:
    app = get_app_from_request()
    ensure_init(app)
    lines = [f"main: cwd={app.bash_state.cwd} state={app.bash_state.state} cmd={app.bash_state.last_command or '(idle)'}"]
    for name, shell in app.get_sessions().items():
        lines.append(f"{name}: cwd={shell.cwd} state={shell.state} cmd={shell.last_command or '(idle)'}")
    for cid, state in app.bash_state.background_shells.items():
        lines.append(f"bg/{cid}: {state.last_command} (state={state.state})")
    return "\n".join(lines)


@mcp.resource("babash://history", description="Command history with success/failure and error hints")
def command_history() -> str:
    app = get_app_from_request()
    history = app.get_history()
    if not history:
        return "No commands executed yet."
    lines = []
    for i, rec in enumerate(history[-20:], 1):
        status = "✓" if rec.success else "✗"
        lines.append(f"{i}. [{status}] [{rec.session}] $ {rec.command}")
        for err in rec.errors:
            lines.append(f"   {err}")
    return "\n".join(lines)


# --- Health ---

@mcp.custom_route("/health", methods=["GET"])  # type: ignore[untyped-decorator]
async def health_check(request: Any) -> Any:
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok", "server": "babash"})


# --- Prompts ---

@mcp.prompt(name="KnowledgeTransfer", description="Save task context for knowledge transfer or resumption.")
async def knowledge_transfer(ctx: McpContext) -> str:  # type: ignore[type-arg]
    return KTS[get_app(ctx).bash_state.mode]


# --- Tools ---

@mcp.tool(
    description="Initialize the shell environment. Optional — auto-initializes on first tool call.",
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
    app = get_app(ctx)
    app.initialized = True
    init_arg = Initialize(type=type, any_workspace_path=any_workspace_path,
                          initial_files_to_read=initial_files_to_read or [],
                          task_id_to_resume=task_id_to_resume, mode_name=mode_name)
    output, _, _ = _handle_initialize(init_arg, make_context(app), CODING_MAX_TOKENS, NONCODING_MAX_TOKENS)
    instructions = f"\n{app.custom_instructions}" if app.custom_instructions else ""
    return f"{output[0]}{instructions}\nInitialize call done.\n"


@mcp.tool(
    description="Execute a shell command. Use session= for parallel execution.",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True),
)
async def run_command(
    ctx: McpContext,  # type: ignore[type-arg]
    command: str,
    is_background: bool = False,
    wait_for_seconds: float | None = None,
    session: str | None = None,
) -> str:
    app = get_app(ctx)
    ensure_init(app)
    shell = app.get_shell(session)

    # Dangerous command elicitation
    dangerous = ["rm -rf", "rm -r /", "mkfs", "dd if=", "> /dev/", ":(){ :|:& };:"]
    if any(p in command for p in dangerous):
        try:
            from pydantic import BaseModel as _BM
            class Confirm(_BM):
                proceed: bool = False
            from mcp.server.elicitation import AcceptedElicitation
            result = await ctx.elicit(f"⚠️ Dangerous command: `{command}`\nProceed?", Confirm)
            if not isinstance(result, AcceptedElicitation) or not result.data.proceed:
                return "Command cancelled by user."
        except Exception:
            pass

    # Multi-line → bash -c
    if "\n" in command.strip():
        command = f"bash -c {shlex.quote(command)}"

    await ctx.info(f"$ {command}")
    await ctx.report_progress(0, 1, "executing...")

    # Busy check
    if shell.state == "pending" and not is_background:
        sname = session or "main"
        return (
            f"Cannot run — session '{sname}' busy.\n"
            f"Running: {shell.last_command or 'unknown'} ({shell.get_pending_for()})\n"
            f"Options: check_status(session='{sname}'), send_keys('Ctrl-c', session='{sname}'), or is_background=true"
        )

    bash_cmd = BashCommand.model_validate({
        "type": "command", "command": command, "is_background": is_background,
        "wait_for_seconds": wait_for_seconds, "thread_id": shell.current_thread_id,
    })
    output, _ = execute_bash(shell, default_enc, bash_cmd, NONCODING_MAX_TOKENS, wait_for_seconds)

    if output.startswith(command.strip()):
        output = output[len(command.strip()):]

    sname = session or "main"
    app.get_last_outputs()[sname] = output
    record_command(app, command, output, sname)

    errors = detect_errors(output)
    if errors:
        output += "\n\n--- Hints ---\n" + "\n".join(errors)

    if not output.strip() or output.strip().startswith("---\n\nstatus"):
        output = "(ok, no output)" if shell.state == "repl" else "(running, no output yet)"

    await ctx.report_progress(1, 1, "done")
    shell.save_state_to_disk()
    return output


@mcp.tool(
    description="Check command status. Returns new output since last check.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
)
async def check_status(
    ctx: McpContext,  # type: ignore[type-arg]
    bg_command_id: str | None = None,
    session: str | None = None,
    wait_for_seconds: float | None = None,
) -> str:
    app = get_app(ctx)
    ensure_init(app)
    shell = app.get_shell(session)

    bash_cmd = BashCommand.model_validate({
        "type": "status_check", "status_check": True, "bg_command_id": bg_command_id,
        "wait_for_seconds": wait_for_seconds, "thread_id": shell.current_thread_id,
    })
    output, _ = execute_bash(shell, default_enc, bash_cmd, NONCODING_MAX_TOKENS, wait_for_seconds)

    sname = session or "main"
    last_outputs = app.get_last_outputs()
    incremental = get_incremental(output, last_outputs.get(sname, ""))
    last_outputs[sname] = output

    if not incremental.strip() or incremental == "(no new output)":
        incremental = f"(no new output)\nstate: {shell.state}\nlast command: {shell.last_command or '(none)'}"

    shell.save_state_to_disk()
    return incremental


@mcp.tool(
    description="Send text input to a running interactive program.",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False),
)
async def send_input(
    ctx: McpContext,  # type: ignore[type-arg]
    text: str, bg_command_id: str | None = None, session: str | None = None,
) -> str:
    app = get_app(ctx)
    ensure_init(app)
    shell = app.get_shell(session)
    bash_cmd = BashCommand.model_validate({
        "type": "send_text", "send_text": text, "bg_command_id": bg_command_id,
        "thread_id": shell.current_thread_id,
    })
    output, _ = execute_bash(shell, default_enc, bash_cmd, NONCODING_MAX_TOKENS, None)
    shell.save_state_to_disk()
    return output


@mcp.tool(
    description="Send special keys. Use Ctrl-c to interrupt, arrow keys to navigate.",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False),
)
async def send_keys(
    ctx: McpContext,  # type: ignore[type-arg]
    keys: list[str] | str = "Ctrl-c", bg_command_id: str | None = None, session: str | None = None,
) -> str:
    app = get_app(ctx)
    ensure_init(app)
    shell = app.get_shell(session)
    keys_list = [keys] if isinstance(keys, str) else keys
    bash_cmd = BashCommand.model_validate({
        "type": "send_specials", "send_specials": keys_list, "bg_command_id": bg_command_id,
        "thread_id": shell.current_thread_id,
    })
    output, _ = execute_bash(shell, default_enc, bash_cmd, NONCODING_MAX_TOKENS, None)
    shell.save_state_to_disk()
    return output


# --- Session tools ---

@mcp.tool(
    description="Create a named shell session for parallel work.",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False),
)
async def create_session(
    ctx: McpContext,  # type: ignore[type-arg]
    name: str, working_directory: str = "",
) -> str:
    app = get_app(ctx)
    ensure_init(app)
    if name == "main":
        return "Error: 'main' already exists."
    sessions = app.get_sessions()
    if name in sessions:
        return f"Session '{name}' already exists."
    cwd = working_directory or app.bash_state.cwd
    sessions[name] = BashState(
        console=app.console, working_dir=cwd, bash_command_mode=None,
        file_edit_mode=None, write_if_empty_mode=None, mode=None,
        use_screen=True, whitelist_for_overwrite=None, thread_id=None,
        shell_path=_shell_path or None,
    )
    return f"Session '{name}' created (cwd: {cwd})."


@mcp.tool(
    description="List all shell sessions and their status.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
)
async def list_sessions(ctx: McpContext) -> str:  # type: ignore[type-arg]
    app = get_app(ctx)
    ensure_init(app)
    lines = [f"- main: cwd={app.bash_state.cwd} state={app.bash_state.state} (default)"]
    for name, shell in app.get_sessions().items():
        lines.append(f"- {name}: cwd={shell.cwd} state={shell.state} cmd={shell.last_command or '(none)'}")
    return "\n".join(lines)


@mcp.tool(
    description="Destroy a named session.",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False),
)
async def destroy_session(ctx: McpContext, name: str) -> str:  # type: ignore[type-arg]
    app = get_app(ctx)
    if name == "main":
        return "Error: cannot destroy main session."
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


# --- File tools ---

@mcp.tool(
    description="Read file contents. Supports line ranges: file.py:10-20",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
)
async def read_files_tool(ctx: McpContext, file_paths: list[str]) -> str:  # type: ignore[type-arg]
    app = get_app(ctx)
    ensure_init(app)
    rf = ReadFiles(file_paths=file_paths)
    result, file_ranges, _ = read_files(
        rf.file_paths, CODING_MAX_TOKENS, NONCODING_MAX_TOKENS, make_context(app),
        rf.start_line_nums, rf.end_line_nums,
    )
    if file_ranges:
        app.bash_state.add_to_whitelist_for_overwrite(file_ranges)
    app.bash_state.save_state_to_disk()
    return result


@mcp.tool(
    description="Read an image file. Provide absolute path.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
)
async def read_image(ctx: McpContext, file_path: str) -> str:  # type: ignore[type-arg]
    app = get_app(ctx)
    ensure_init(app)
    image = read_image_from_shell(file_path, make_context(app))
    return f"[Image: {image.media_type}, {len(image.data)} bytes base64]"


@mcp.tool(
    description="Create a new file. Fails if file exists.",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False),
)
async def create_file(ctx: McpContext, file_path: str, content: str) -> str:  # type: ignore[type-arg]
    app = get_app(ctx)
    ensure_init(app)
    wf = WriteIfEmpty(file_path=file_path, file_content=content)
    result, paths = write_file(wf, True, CODING_MAX_TOKENS, NONCODING_MAX_TOKENS, make_context(app))
    if paths:
        app.bash_state.add_to_whitelist_for_overwrite(paths)
    app.bash_state.save_state_to_disk()
    return result


with open(os.path.join(os.path.dirname(__file__), "..", "diff-instructions.txt")) as _f:
    _diff_instructions = _f.read()


@mcp.tool(
    description="Edit an existing file.\n" + _diff_instructions,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False),
)
async def file_write_or_edit(
    ctx: McpContext,  # type: ignore[type-arg]
    file_path: str, percentage_to_change: int, text_or_search_replace_blocks: str,
) -> str:
    app = get_app(ctx)
    ensure_init(app)
    fwe = FileWriteOrEdit(
        file_path=file_path, percentage_to_change=percentage_to_change,
        text_or_search_replace_blocks=text_or_search_replace_blocks,
        thread_id=app.bash_state.current_thread_id,
    )
    result, paths = file_writing(fwe, CODING_MAX_TOKENS, NONCODING_MAX_TOKENS, make_context(app))
    if paths:
        app.bash_state.add_to_whitelist_for_overwrite(paths)
    app.bash_state.save_state_to_disk()
    return result


@mcp.tool(
    description="Save task context for later resumption.",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False),
)
async def context_save(
    ctx: McpContext,  # type: ignore[type-arg]
    id: str, description: str, relevant_file_globs: list[str], project_root_path: str = "",
) -> str:
    app = get_app(ctx)
    ensure_init(app)
    cs = ContextSave(id=id, project_root_path=project_root_path,
                     description=description, relevant_file_globs=relevant_file_globs)
    return _handle_context_save(cs, make_context(app))


# --- Entry point ---

async def main(shell_path: str = "") -> None:
    global _shell_path
    _shell_path = shell_path
    await mcp.run_stdio_async()
