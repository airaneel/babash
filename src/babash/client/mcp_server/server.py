"""babash MCP server — wiring only. All logic lives in helpers/state/tools."""

import anyio
import base64
import logging
import os
import shlex
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import metadata
from typing import Any, Literal

from mcp.server.fastmcp import Context as McpContext
from mcp.server.fastmcp import FastMCP
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
from ..file_ops.search_replace import SEARCH_MARKER, search_replace_edit
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
from .helpers import (
    CODING_MAX_TOKENS,
    NONCODING_MAX_TOKENS,
    detect_errors,
    record_command,
)
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
        console=console,
        working_dir=os.path.join(tmp_dir, "claude_playground"),
        bash_command_mode=None,
        file_edit_mode=None,
        write_if_empty_mode=None,
        mode=None,
        use_screen=True,
        whitelist_for_overwrite=None,
        thread_id=None,
        shell_path=_shell_path or None,
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
- run_command(command): execute a command and get output. Returns quickly. If the
  command is still running you get a "pending" status — that's normal, use
  check_status to poll or work in another session in the meantime.
- check_status(wait_for_seconds=N): get new output since the last check. N is capped at 5s server-side.
  For long-running commands (ansible, builds), space out your checks — read the incremental
  output each time and do other work in between. Don't call it in a tight loop expecting it to block.
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

# Background commands
run_command(command="npm start", is_background=true) auto-creates a session (e.g. "bg_a1b2c3").
Use check_status(session="bg_a1b2c3") to monitor, send_keys("Ctrl-c", session="bg_a1b2c3") to stop.
Background sessions appear in list_sessions() and can be destroyed with destroy_session().

# File operations
- read_files_tool(file_paths): read files. Supports line ranges: file.py:10-20
- create_file(file_path, content): create a new file (fails if exists)
- file_write_or_edit(file_path, percentage_to_change, text_or_search_replace_blocks): edit existing files

All file tools accept session= to operate on files in a remote session (e.g. one running SSH):
  read_files_tool(file_paths=["/etc/hosts"], session="myserver")
  create_file(file_path="/tmp/test.txt", content="hello", session="myserver")
  file_write_or_edit(file_path="/etc/config.yaml", ..., session="myserver")

Do NOT use echo/cat/sed to read or write files — use the file tools instead.

# Important
- Each session runs one foreground command at a time.
- If a command is still running, check_status or send_keys(Ctrl-c) before running another.
- Do NOT poll check_status repeatedly waiting for a command to finish. If the command has no output
  after one check, either send_keys(Ctrl-c) and try a different approach, or move on to other work
  in a different session. Never call check_status more than 2-3 times for the same command.
- cd, env vars, and state persist within each session independently.
- If output is truncated, use more precise commands (grep, head, tail, awk) instead of dumping everything.
- For large files, use read_files_tool with line ranges (file.py:1-50) instead of reading the whole file.
- For SSH: open an interactive session with run_command("ssh user@host"), then run commands directly.
  Do NOT use ssh user@host "cmd" repeatedly — it reconnects and re-authenticates every time.
  Use a session for long SSH work: create_session("remote"), run_command("ssh user@host", session="remote").
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
    initialize(
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


def make_context(app: AppState) -> Context:
    return Context(app.bash_state, app.console)


def _exec_in_session(shell: BashState, command: str) -> str:
    """Run a single-line command through a session shell and return output."""
    bash_cmd = BashCommand.model_validate({
        "type": "command",
        "command": command,
        "is_background": False,
        "wait_for_seconds": None,
        "thread_id": shell.current_thread_id,
    })
    output, _ = execute_bash(shell, default_enc, bash_cmd, NONCODING_MAX_TOKENS, None)
    return output


def _session_read_file(shell: BashState, path: str, session_name: str) -> str:
    """Read a file through a session shell using cat -n."""
    if shell.state == "pending":
        return (
            f"Error: session '{session_name}' is busy "
            f"(running: {shell.last_command or 'unknown'}).\n"
            f"Use check_status(session='{session_name}') or "
            f"send_keys('Ctrl-c', session='{session_name}') first."
        )
    output = _exec_in_session(shell, f"cat -n {shlex.quote(path)}")
    return output


def _session_write_file(shell: BashState, path: str, content: str, session_name: str) -> str:
    """Write a file through a session shell using base64."""
    if shell.state == "pending":
        return (
            f"Error: session '{session_name}' is busy "
            f"(running: {shell.last_command or 'unknown'}).\n"
            f"Use check_status(session='{session_name}') or "
            f"send_keys('Ctrl-c', session='{session_name}') first."
        )
    encoded = base64.b64encode(content.encode()).decode()
    parent = os.path.dirname(path)
    if parent:
        _exec_in_session(shell, f"mkdir -p {shlex.quote(parent)}")
    output = _exec_in_session(
        shell,
        f"printf '%s' '{encoded}' | base64 -d > {shlex.quote(path)}",
    )
    # Verify write
    verify = _exec_in_session(shell, f"wc -c < {shlex.quote(path)}")
    expected = len(content.encode())
    return output or f"Success (wrote {expected} bytes to {path})\n{verify}"


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
    for tool in [
        "git",
        "docker",
        "python3",
        "node",
        "npm",
        "uv",
        "pip",
        "rg",
        "jq",
        "ssh",
        "curl",
    ]:
        path = shutil.which(tool)
        if path:
            lines.append(f"has_{tool}: {path}")
    return "\n".join(lines)


@mcp.resource(
    "babash://workspace/processes", description="All sessions and running commands"
)
def workspace_processes() -> str:
    app = get_app_from_request()
    ensure_init(app)
    lines = [
        f"main: cwd={app.bash_state.cwd} state={app.bash_state.state} cmd={app.bash_state.last_command or '(idle)'}"
    ]
    for name, shell in app.get_sessions().items():
        lines.append(
            f"{name}: cwd={shell.cwd} state={shell.state} cmd={shell.last_command or '(idle)'}"
        )
    for cid, state in app.bash_state.background_shells.items():
        lines.append(f"bg/{cid}: {state.last_command} (state={state.state})")
    return "\n".join(lines)


@mcp.resource(
    "babash://history",
    description="Command history with success/failure and error hints",
)
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


@mcp.prompt(
    name="KnowledgeTransfer",
    description="Save task context for knowledge transfer or resumption.",
)
async def knowledge_transfer(ctx: McpContext) -> str:  # type: ignore[type-arg]
    return KTS[get_app(ctx).bash_state.mode]


# --- Tools ---


@mcp.tool(
    description="Initialize the shell environment. Optional — auto-initializes on first tool call.",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def babash_initialize(
    ctx: McpContext,  # type: ignore[type-arg]
    type: Literal[
        "first_call",
        "user_asked_mode_change",
        "reset_shell",
        "user_asked_change_workspace",
    ] = "first_call",
    any_workspace_path: str = "",
    initial_files_to_read: list[str] | None = None,
    task_id_to_resume: str = "",
    mode_name: Literal["babash", "architect", "code_writer"] = "babash",
) -> str:
    app = get_app(ctx)
    app.initialized = True
    init_arg = Initialize(
        type=type,
        any_workspace_path=any_workspace_path,
        initial_files_to_read=initial_files_to_read or [],
        task_id_to_resume=task_id_to_resume,
        mode_name=mode_name,
    )
    output, _, _ = _handle_initialize(
        init_arg, make_context(app), CODING_MAX_TOKENS, NONCODING_MAX_TOKENS
    )
    instructions = f"\n{app.custom_instructions}" if app.custom_instructions else ""
    return f"{output[0]}{instructions}\nInitialize call done.\n"


@mcp.tool(
    description=(
        "Execute a shell command. Long commands return immediately with state=pending — "
        "never use `sleep` to wait, call check_status instead. For parallel work use "
        "session= (named) or is_background=True (auto-named bg_*). To create or edit "
        "files prefer create_file / file_write_or_edit over `cat <<EOF` — they preserve "
        "quoting and work for remote sessions. Multi-line commands run in a subshell, so "
        "cd/exports inside them do NOT persist to the session — use single-line commands "
        "to change session state."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def run_command(
    ctx: McpContext,  # type: ignore[type-arg]
    command: str,
    is_background: bool = False,
    session: str | None = None,
) -> str:
    app = get_app(ctx)
    ensure_init(app)
    shell = app.get_shell(session)

    # No in-server destructive-command gate: this client (Claude Desktop)
    # doesn't support MCP elicitation, so any prompt here would silently
    # no-op and give a false sense of safety. Destructive actions are gated
    # by the host agent's own guardrails instead.

    # Multi-line → bash -c
    if "\n" in command.strip():
        command = f"bash -c {shlex.quote(command)}"

    await ctx.info(f"$ {command}")
    await ctx.report_progress(0, 1, "executing...")

    sname = session or "main"

    # is_background → auto-create a session
    if is_background:
        import hashlib

        # Key on cwd+command so the same command in two dirs gets two sessions.
        bg_key = f"{shell.cwd}\0{command}"
        bg_name = f"bg_{hashlib.md5(bg_key.encode()).hexdigest()[:6]}"
        if bg_name not in app.get_sessions():
            cwd = shell.cwd
            new_shell = BashState(
                console=app.console, working_dir=cwd,
                bash_command_mode=None, file_edit_mode=None,
                write_if_empty_mode=None, mode=None,
                use_screen=True, whitelist_for_overwrite=None,
                thread_id=None, shell_path=_shell_path or None,
            )
            app.get_sessions()[bg_name] = new_shell
        shell = app.get_sessions()[bg_name]
        sname = bg_name

    # Busy check: don't error — run a status check instead so the agent gets
    # forward progress (incremental output + current state) instead of retrying
    # the same command in a loop.
    if shell.state == "pending":
        status_cmd = BashCommand.model_validate({
            "type": "status_check",
            "status_check": True,
            "wait_for_seconds": None,
            "thread_id": shell.current_thread_id,
        })
        status_out, _ = await anyio.to_thread.run_sync(
            execute_bash, shell, default_enc, status_cmd, NONCODING_MAX_TOKENS, None
        )
        new_text, _, _ = status_out.partition("\n\n---\n\n")
        new_text = new_text.strip()
        last = app.get_last_outputs().get(sname, "")
        if last and new_text.startswith(last):
            new_text = new_text[len(last):].lstrip()
        app.get_last_outputs()[sname] = status_out
        shell.save_state_to_disk()
        header = (
            f"Session '{sname}' is still running '{shell.last_command or 'unknown'}' "
            f"for {shell.get_pending_for()}. Cannot start a new command until it "
            f"finishes. Use send_keys('Ctrl-c', session='{sname}') to interrupt, "
            f"check_status(session='{sname}') to wait, or create_session(name='other') "
            f"to run in parallel."
        )
        if new_text:
            return f"{header}\n\n--- new output ---\n{new_text}"
        return f"{header}\n\n(no new output yet)"

    bash_cmd = BashCommand.model_validate(
        {
            "type": "command",
            "command": command,
            "is_background": False,
            "wait_for_seconds": None,
            "thread_id": shell.current_thread_id,
        }
    )
    output, _ = await anyio.to_thread.run_sync(
        execute_bash, shell, default_enc, bash_cmd, NONCODING_MAX_TOKENS, None
    )

    if output.startswith(command.strip()):
        output = output[len(command.strip()) :]

    if is_background:
        output = f"Running in session '{sname}'. Use check_status(session='{sname}') to monitor.\n{output}"
    app.get_last_outputs()[sname] = output
    record_command(app, command, output, sname)

    errors = detect_errors(output)
    if errors:
        output += "\n\n--- Hints ---\n" + "\n".join(errors)

    if not output.strip() or output.strip().startswith("---\n\nstatus"):
        output = (
            "(ok, no output)" if shell.state == "repl" else "(running, no output yet)"
        )

    await ctx.report_progress(1, 1, "done")
    shell.save_state_to_disk()
    return output


@mcp.tool(
    description=(
        "Check a running command's status and return new output since the last check. "
        "Use this instead of `sleep` when waiting — pass wait_for_seconds to block up to "
        "5s per call (hard cap); for longer waits, call again."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def check_status(
    ctx: McpContext,  # type: ignore[type-arg]
    session: str | None = None,
    wait_for_seconds: float | None = None,
) -> str:
    app = get_app(ctx)
    ensure_init(app)
    shell = app.get_shell(session)

    # Hard cap: never block a session for more than 5s on a single check.
    # If the command needs longer, the agent calls check_status again.
    capped_wait = (
        min(float(wait_for_seconds), 5.0) if wait_for_seconds else None
    )
    bash_cmd = BashCommand.model_validate(
        {
            "type": "status_check",
            "status_check": True,
            "wait_for_seconds": capped_wait,
            "thread_id": shell.current_thread_id,
        }
    )
    output, _ = await anyio.to_thread.run_sync(
        execute_bash, shell, default_enc, bash_cmd, NONCODING_MAX_TOKENS, capped_wait
    )

    # execute_bash returns "<new pending text>\n\n---\n\nstatus = ...\ncwd = ..."
    # The "---" gets rendered as a markdown horizontal rule in the UI, hiding
    # everything after. Split and format as a plain status line instead.
    new_text, _, _ = output.partition("\n\n---\n\n")
    new_text = new_text.strip()

    sname = session or "main"
    last_outputs = app.get_last_outputs()
    last = last_outputs.get(sname, "")
    if last and new_text.startswith(last):
        new_text = new_text[len(last):].lstrip()
    last_outputs[sname] = output

    status_line = f"[state={shell.state} cwd={shell.cwd}"
    if shell.state == "pending":
        status_line += f" running={shell.last_command or '(unknown)'} for={shell.get_pending_for()}"
    status_line += "]"

    if new_text:
        result = f"{new_text}\n\n{status_line}"
    else:
        result = f"(no new output) {status_line}"

    shell.save_state_to_disk()
    return result


@mcp.tool(
    description="Send text input to a running program (passwords, prompts). For Enter/Ctrl-c use send_keys instead.",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
async def send_input(
    ctx: McpContext,  # type: ignore[type-arg]
    text: str,
    session: str | None = None,
) -> str:
    app = get_app(ctx)
    ensure_init(app)
    shell = app.get_shell(session)

    if not text:
        return "Error: text cannot be empty. Use send_keys('Enter') to press Enter, or send_keys('Ctrl-c') to interrupt."

    bash_cmd = BashCommand.model_validate({
        "type": "send_text",
        "send_text": text,
        "thread_id": shell.current_thread_id,
    })
    output, _ = await anyio.to_thread.run_sync(
        execute_bash, shell, default_enc, bash_cmd, NONCODING_MAX_TOKENS, None
    )
    shell.save_state_to_disk()
    return output


@mcp.tool(
    description=(
        "Send special keys. Use Ctrl-c to interrupt, arrow keys to navigate. "
        "babash has no built-in command timeout and macOS has no `timeout` binary "
        "without coreutils — if a run_command hangs, kill it with Ctrl-c here."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
async def send_keys(
    ctx: McpContext,  # type: ignore[type-arg]
    keys: list[str] | str = "Ctrl-c",
    session: str | None = None,
) -> str:
    app = get_app(ctx)
    ensure_init(app)
    shell = app.get_shell(session)
    keys_list = [keys] if isinstance(keys, str) else keys
    bash_cmd = BashCommand.model_validate(
        {
            "type": "send_specials",
            "send_specials": keys_list,
            "thread_id": shell.current_thread_id,
        }
    )
    output, _ = await anyio.to_thread.run_sync(
        execute_bash, shell, default_enc, bash_cmd, NONCODING_MAX_TOKENS, None
    )
    shell.save_state_to_disk()
    return output


# --- Session tools ---


@mcp.tool(
    description="Create a named shell session for parallel work.",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
async def create_session(
    ctx: McpContext,  # type: ignore[type-arg]
    name: str,
    working_directory: str = "",
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
    return f"Session '{name}' created (cwd: {cwd})."


@mcp.tool(
    description="List all shell sessions and their status.",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def list_sessions(ctx: McpContext) -> str:  # type: ignore[type-arg]
    app = get_app(ctx)
    ensure_init(app)
    lines = [f"- main: cwd={app.bash_state.cwd} state={app.bash_state.state} (default)"]
    for name, shell in app.get_sessions().items():
        lines.append(
            f"- {name}: cwd={shell.cwd} state={shell.state} cmd={shell.last_command or '(none)'}"
        )
    return "\n".join(lines)


@mcp.tool(
    description="Destroy a named session.",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    ),
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
    description="Read file contents. Supports line ranges: file.py:10-20. Use session= to read from a remote session (e.g. one running SSH).",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def read_files_tool(ctx: McpContext, file_paths: list[str], session: str | None = None) -> str:  # type: ignore[type-arg]
    app = get_app(ctx)
    ensure_init(app)

    if session:
        shell = app.get_shell(session)
        if shell.state == "pending":
            return (
                f"Error: session '{session}' is busy "
                f"(running: {shell.last_command or 'unknown'}).\n"
                f"Use check_status(session='{session}') or "
                f"send_keys('Ctrl-c', session='{session}') first."
            )
        rf = ReadFiles(file_paths=file_paths)
        message = ""
        for i, path in enumerate(rf.file_paths):
            start = rf.start_line_nums[i]
            end = rf.end_line_nums[i]
            if start is not None or end is not None:
                s = start or 1
                e_cond = f" && NR<={end}" if end else ""
                cmd = f"awk 'NR>={s}{e_cond} {{printf \"%d %s\\n\", NR, $0}}' {shlex.quote(path)}"
            else:
                cmd = f"awk '{{printf \"%d %s\\n\", NR, $0}}' {shlex.quote(path)}"
            output = _exec_in_session(shell, cmd)
            range_str = ""
            if start or end:
                range_str = f":{start or ''}-{end or ''}"
            message += f'\n<file-contents-numbered path="{path}{range_str}">\n{output}\n</file-contents-numbered>'
        return message or "(no files)"

    rf = ReadFiles(file_paths=file_paths)
    result, file_ranges, _ = read_files(
        rf.file_paths,
        CODING_MAX_TOKENS,
        NONCODING_MAX_TOKENS,
        make_context(app),
        rf.start_line_nums,
        rf.end_line_nums,
    )
    if file_ranges:
        app.bash_state.add_to_whitelist_for_overwrite(file_ranges)
    app.bash_state.save_state_to_disk()
    return result


@mcp.tool(
    description="Read an image file. Provide absolute path.",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def read_image(ctx: McpContext, file_path: str) -> str:  # type: ignore[type-arg]
    app = get_app(ctx)
    ensure_init(app)
    image = read_image_from_shell(file_path, make_context(app))
    return f"[Image: {image.media_type}, {len(image.data)} bytes base64]"


@mcp.tool(
    description=(
        "Create a new file. Fails if file exists. Prefer this over `cat <<EOF` in "
        "run_command — content is transferred without shell quoting, so YAML/Jinja/"
        "quotes survive verbatim. Pass session= to write into a remote SSH session."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
async def create_file(ctx: McpContext, file_path: str, content: str, session: str | None = None) -> str:  # type: ignore[type-arg]
    app = get_app(ctx)
    ensure_init(app)

    if session:
        shell = app.get_shell(session)
        # Check if file exists
        check = _exec_in_session(shell, f"test -f {shlex.quote(file_path)} && echo EXISTS || echo NO")
        if "EXISTS" in check:
            return f"Error: file {file_path} already exists."
        return _session_write_file(shell, file_path, content, session)

    wf = WriteIfEmpty(file_path=file_path, file_content=content)
    result, paths = write_file(
        wf, True, CODING_MAX_TOKENS, NONCODING_MAX_TOKENS, make_context(app)
    )
    if paths:
        app.bash_state.add_to_whitelist_for_overwrite(paths)
    app.bash_state.save_state_to_disk()
    return result


with open(os.path.join(os.path.dirname(__file__), "..", "diff-instructions.txt")) as _f:
    _diff_instructions = _f.read()


@mcp.tool(
    description=(
        "Edit an existing file (or write it whole). Prefer this over `cat <<EOF` / "
        "`sed` / `echo >>` in run_command — content is transferred without shell "
        "quoting, so YAML/Jinja/quotes survive verbatim. Pass session= to edit "
        "inside a remote SSH session.\n"
    ) + _diff_instructions,
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def file_write_or_edit(
    ctx: McpContext,  # type: ignore[type-arg]
    file_path: str,
    percentage_to_change: int,
    text_or_search_replace_blocks: str,
    session: str | None = None,
) -> str:
    app = get_app(ctx)
    ensure_init(app)

    if session:
        shell = app.get_shell(session)
        if shell.state == "pending":
            return (
                f"Error: session '{session}' is busy "
                f"(running: {shell.last_command or 'unknown'})."
            )
        content = text_or_search_replace_blocks.strip()
        edit_lines = content.split("\n")
        is_edit = bool(SEARCH_MARKER.match(edit_lines[0])) or (0 < percentage_to_change <= 50)

        if is_edit:
            # Read current file, apply search-replace, write back
            raw = _exec_in_session(shell, f"cat {shlex.quote(file_path)}")
            if not raw.strip():
                check = _exec_in_session(shell, f"test -f {shlex.quote(file_path)} && echo EXISTS || echo NO")
                if "NO" in check:
                    return f"Error: file {file_path} does not exist"
            new_content, comments = search_replace_edit(edit_lines, raw, app.console.log)
            result = _session_write_file(shell, file_path, new_content, session)
            return f"{comments}\n{result}" if comments else result
        else:
            # Full write
            return _session_write_file(shell, file_path, content, session)

    fwe = FileWriteOrEdit(
        file_path=file_path,
        percentage_to_change=percentage_to_change,
        text_or_search_replace_blocks=text_or_search_replace_blocks,
        thread_id=app.bash_state.current_thread_id,
    )
    result, paths = file_writing(
        fwe, CODING_MAX_TOKENS, NONCODING_MAX_TOKENS, make_context(app)
    )
    if paths:
        app.bash_state.add_to_whitelist_for_overwrite(paths)
    app.bash_state.save_state_to_disk()
    return result


@mcp.tool(
    description="Save task context for later resumption.",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def context_save(
    ctx: McpContext,  # type: ignore[type-arg]
    id: str,
    description: str,
    relevant_file_globs: list[str],
    project_root_path: str = "",
) -> str:
    app = get_app(ctx)
    ensure_init(app)
    cs = ContextSave(
        id=id,
        project_root_path=project_root_path,
        description=description,
        relevant_file_globs=relevant_file_globs,
    )
    return _handle_context_save(cs, make_context(app))


# --- Entry point ---


async def main(shell_path: str = "") -> None:
    global _shell_path
    _shell_path = shell_path
    await mcp.run_stdio_async()
