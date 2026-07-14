"""The shell tools: initialize, run, poll, and type into a session."""

import hashlib
import shlex
from dataclasses import dataclass

import anyio
from mcp.server.fastmcp import Context as McpContext
from mcp.types import ToolAnnotations

from ....types_ import Command, SendSpecials, SendText, Specials, StatusCheck
from ...bash_state import STATUS_SEPARATOR, BashState, execute_bash
from ..chat import new_chat_id, resolve_chat, roster_footer, warmup_shell
from ..helpers import detect_errors, record_command
from ..instance import get_app, mcp
from ..state import AppState, ChatWorkspace

# A single check_status never blocks a session for longer than this. If the
# command needs more, the agent calls again — which keeps it able to react.
MAX_WAIT_SECONDS = 5.0

# How long a background command is watched before we hand control back. It is
# short on purpose: is_background=True is the agent saying it will not wait, so
# spending the full default budget on it is pure latency. What this does buy is
# the common case of a command that dies immediately — a typo, a missing binary
# — which is worth reporting now rather than making the agent poll to discover.
BACKGROUND_GRACE_SECONDS = 0.5


@dataclass(frozen=True)
class _Target:
    """Where a command runs, and everything that follows from that.

    Whether a command is "background" changes three things at once: which shell
    it goes to, how long we watch it, and whether the reply has to say where it
    went. Deciding all three here means the rest of run_command never has to ask
    the question again.
    """

    shell: BashState
    name: str
    budget: float | None
    preamble: str


def _background_name(chat: ChatWorkspace, shell: BashState, command: str) -> str:
    """Derived from cwd+command, so re-running the same thing lands back in the
    same session rather than leaking a fresh shell on every invocation.

    But if that session is still busy with the previous run, the agent asking
    again means it wants a *second* copy running alongside — N workers, N builds
    of different branches — not to be told the name is taken. So a busy name
    steps aside for a numbered one.
    """
    key = "\0".join([shell.cwd, command])
    base = f"bg_{hashlib.md5(key.encode()).hexdigest()[:6]}"
    name, n = base, 2
    while name in chat.sessions and chat.sessions[name].state == "pending":
        name, n = f"{base}_{n}", n + 1
    return name


async def _target(
    app: AppState,
    chat: ChatWorkspace,
    shell: BashState,
    session: str | None,
    command: str,
    is_background: bool,
) -> _Target:
    if not is_background:
        return _Target(shell=shell, name=session or "main", budget=None, preamble="")

    name = _background_name(chat, shell, command)
    if name not in chat.sessions:
        chat.sessions[name] = app.new_shell(shell.cwd)
        await warmup_shell(chat.sessions[name])
    return _Target(
        shell=chat.sessions[name],
        name=name,
        budget=BACKGROUND_GRACE_SECONDS,
        preamble=(
            f"Running in session '{name}'. "
            f"Use check_status(session='{name}') to monitor.\n"
        ),
    )


@mcp.tool(
    description=(
        "REQUIRED first call in every conversation. Creates an isolated shell "
        "workspace for this chat and returns a chat_id. You MUST pass that same "
        "chat_id to every other babash tool for the rest of the conversation — "
        "it is how the server keeps your shell separate from other chats sharing "
        "the same server. Leave chat_id empty to be assigned a fresh one; pass an "
        "existing chat_id to reset that workspace and start its shell over."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
async def babash_initialize(
    chat_id: str = "",
    working_directory: str = "",
) -> str:
    app = get_app()
    cid = chat_id or new_chat_id()
    # create_chat tears down any existing workspace for this id, so passing a
    # known chat_id is how you reset a wedged shell.
    chat = app.create_chat(cid)
    if working_directory:
        await anyio.to_thread.run_sync(
            execute_bash,
            chat.main,
            Command(command=f"cd {shlex.quote(working_directory)}"),
            app.settings.max_output_chars,
            None,
        )
    await warmup_shell(chat.main)

    # No custom-instructions blob is appended here. MCP already has a channel for
    # telling a model how to use a server — the `instructions` field, which this
    # server fills in — and a second one, stapled to a tool's reply, only means
    # two places to keep in step.
    return (
        f"Your chat_id is: {cid}\n"
        f"IMPORTANT: pass chat_id='{cid}' to EVERY babash tool call for the rest "
        f"of this conversation so your shell stays isolated from other chats.\n\n"
        f"{roster_footer(chat)}"
    )


def _new_since_last(last_output: dict[str, str], sname: str, output: str) -> str:
    """The part of `output` the agent hasn't already been shown for this session.

    execute_bash's reply is "<output>---<status>"; only the output half is worth
    diffing, and the status half gets rebuilt by the caller anyway.
    """
    new_text, _, _ = output.partition(STATUS_SEPARATOR)
    new_text = new_text.strip()
    last = last_output.get(sname, "")
    if last and new_text.startswith(last):
        new_text = new_text[len(last) :].lstrip()
    return new_text


@mcp.tool(
    description=(
        "Execute a shell command. Long commands return immediately with state=pending — "
        "never use `sleep` to wait, call check_status instead. For parallel work use "
        "session= (named) or is_background=True (auto-named bg_*). To create or edit "
        "files use write_file / edit_file rather than `cat <<EOF` — they preserve "
        "quoting and work for remote sessions. Multi-line commands run in a subshell, "
        "so cd/exports inside them do NOT persist to the session — use single-line "
        "commands to change session state."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def run_command(
    ctx: McpContext,  # type: ignore[type-arg]  # for progress/log, not for get_app
    command: str,
    chat_id: str,
    is_background: bool = False,
    session: str | None = None,
) -> str:
    app = get_app()
    chat, err = resolve_chat(app, chat_id)
    if chat is None:
        return err
    try:
        shell = chat.get_shell(session)
    except ValueError as e:
        return f"Error: {e}"

    # No in-server destructive-command gate: this client doesn't support MCP
    # elicitation, so a prompt here would silently no-op and give a false sense
    # of safety. Destructive actions are gated by the host agent's guardrails.

    if "\n" in command.strip():
        command = f"bash -c {shlex.quote(command)}"

    await ctx.info(f"$ {command}")
    await ctx.report_progress(0, 1, "executing...")

    # Decide once where this runs and on what terms, so that "is it background?"
    # is not re-asked at every step below.
    target = await _target(app, chat, shell, session, command, is_background)
    shell, sname = target.shell, target.name

    # Busy session: don't error — do a status check instead, so the agent gets
    # forward progress (new output + current state) rather than being tempted to
    # retry the same command in a loop.
    if shell.state == "pending":
        status_out = await anyio.to_thread.run_sync(
            execute_bash, shell, StatusCheck(), app.settings.max_output_chars, None
        )
        new_text = _new_since_last(chat.last_output, sname, status_out)
        chat.last_output[sname] = status_out
        header = (
            f"Session '{sname}' is still running '{shell.last_command or 'unknown'}' "
            f"for {shell.get_pending_for()}. Cannot start a new command until it "
            f"finishes. Use send_keys('Ctrl-c', session='{sname}') to interrupt, "
            f"check_status(session='{sname}') to wait, or create_session(name='other') "
            f"to run in parallel."
        )
        body = (
            f"{header}\n\n--- new output ---\n{new_text}"
            if new_text
            else f"{header}\n\n(no new output yet)"
        )
        return f"{body}\n\n{roster_footer(chat)}"

    output = target.preamble + await anyio.to_thread.run_sync(
        execute_bash,
        shell,
        Command(command=command),
        app.settings.max_output_chars,
        target.budget,
    )
    chat.last_output[sname] = output
    record_command(chat, command, output, sname)

    errors = detect_errors(output)
    if errors:
        output += "\n\n--- Hints ---\n" + "\n".join(errors)

    if not output.strip() or output.strip().startswith("---\n\nstatus"):
        output = "(ok, no output)" if shell.state == "repl" else "(running, no output yet)"

    await ctx.report_progress(1, 1, "done")
    return f"{output}\n\n{roster_footer(chat)}"


@mcp.tool(
    description=(
        "Check a running command's status and return new output since the last check. "
        "Use this instead of `sleep` when waiting — pass wait_for_seconds to block up "
        "to 5s per call (hard cap); for longer waits, call again."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def check_status(
    chat_id: str,
    session: str | None = None,
    wait_for_seconds: float | None = None,
) -> str:
    app = get_app()
    chat, err = resolve_chat(app, chat_id)
    if chat is None:
        return err
    try:
        shell = chat.get_shell(session)
    except ValueError as e:
        return f"Error: {e}"

    capped_wait = min(float(wait_for_seconds), MAX_WAIT_SECONDS) if wait_for_seconds else None
    output = await anyio.to_thread.run_sync(
        execute_bash, shell, StatusCheck(), app.settings.max_output_chars, capped_wait
    )

    sname = session or "main"
    new_text = _new_since_last(chat.last_output, sname, output)
    chat.last_output[sname] = output

    # execute_bash's own status block starts with "---", which renders as a
    # markdown rule and hides everything after it. Restate it as a plain line.
    prompt = shell.pending_prompt()
    status_line = f"[state={shell.state} cwd={shell.cwd}"
    if shell.state == "pending":
        status_line += (
            f" running={shell.last_command or '(unknown)'} for={shell.get_pending_for()}"
        )
    elif shell.last_exit_code is not None:
        status_line += f" exit={shell.last_exit_code}"
    status_line += "]"

    if prompt:
        # Without this the agent reads "(no new output), still pending" and polls
        # again — but nothing will ever arrive, because the command is waiting on
        # *it*. The stall is not the command's; it's a question.
        head = new_text or "(no new output)"
        result = (
            f"{head}\n\nWAITING FOR INPUT — the command is blocked on a prompt:\n"
            f"  {prompt!r}\n"
            f"Answer with send_input(text=..., session={sname!r}), or send_keys "
            f"for control keys. Polling check_status again will not advance it.\n\n"
            f"{status_line}"
        )
    elif new_text:
        result = f"{new_text}\n\n{status_line}"
    else:
        result = f"(no new output) {status_line}"
    return f"{result}\n\n{roster_footer(chat)}"


@mcp.tool(
    description=(
        "Send text input to a running program (passwords, prompts). "
        "For Enter/Ctrl-c use send_keys instead."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
async def send_input(
    text: str,
    chat_id: str,
    session: str | None = None,
) -> str:
    app = get_app()
    chat, err = resolve_chat(app, chat_id)
    if chat is None:
        return err
    try:
        shell = chat.get_shell(session)
    except ValueError as e:
        return f"Error: {e}"

    if not text:
        return (
            "Error: text cannot be empty. Use send_keys('Enter') to press Enter, "
            "or send_keys('Ctrl-c') to interrupt."
        )

    output = await anyio.to_thread.run_sync(
        execute_bash, shell, SendText(send_text=text), app.settings.max_output_chars, None
    )
    return f"{output}\n\n{roster_footer(chat)}"


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
    chat_id: str,
    keys: list[Specials] | Specials = "Ctrl-c",
    session: str | None = None,
) -> str:
    # `Specials` is a Literal, so FastMCP puts the whole key vocabulary into this
    # tool's JSON Schema as an enum, and pydantic rejects anything outside it
    # before we are called. This used to take a bare `list[str]` and then check
    # the values by hand against `get_args(Specials)` — reimplementing, badly and
    # invisibly to the client, what the type annotation already says. The model
    # can now *see* which keys exist instead of guessing and being corrected.
    app = get_app()
    chat, err = resolve_chat(app, chat_id)
    if chat is None:
        return err
    try:
        shell = chat.get_shell(session)
    except ValueError as e:
        return f"Error: {e}"

    raw_keys = [keys] if isinstance(keys, str) else keys

    output = await anyio.to_thread.run_sync(
        execute_bash,
        shell,
        SendSpecials(send_specials=tuple(raw_keys)),
        app.settings.max_output_chars,
        None,
    )
    return f"{output}\n\n{roster_footer(chat)}"
