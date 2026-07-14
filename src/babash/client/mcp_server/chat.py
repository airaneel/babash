"""Resolving a chat_id to its workspace, and reporting that workspace back."""

from uuid import uuid4

import anyio
import pexpect

from ...types_ import Command
from ..bash_state import BashState, execute_bash
from .state import AppState, ChatWorkspace


def new_chat_id() -> str:
    return uuid4().hex[:12]


def resolve_chat(app: AppState, chat_id: str) -> tuple[ChatWorkspace | None, str]:
    """Look up a chat's workspace, or return a message telling the model how to
    get a valid chat_id.

    Isolation depends on the model passing back the chat_id it was handed by
    babash_initialize, so an unknown id is a usage error — not a reason to
    silently fall back to some other chat's shell.
    """
    chat = app.get_chat(chat_id)
    if chat is not None:
        return chat, ""
    known = ", ".join(app.chats.keys()) or "(none yet)"
    return None, (
        f"Error: unknown chat_id '{chat_id}'. Call babash_initialize once at the "
        f"start of this conversation to get your chat_id, then pass that same "
        f"chat_id to every babash tool call so your shell stays isolated from "
        f"other chats. Known chat_ids: {known}."
    )


def abbreviate(command: str) -> str:
    """A command shortened to something a footer can carry without being a wall."""
    line = " ".join(command.split())
    return line if len(line) <= 60 else line[:57] + "..."


def _activity(shell: BashState, name: str) -> str:
    if shell.state != "pending":
        return "idle"
    prompt = shell.pending_prompt()
    if prompt:
        return f"WAITING FOR INPUT at {prompt!r} — send_input(text=..., session={name!r})"
    return f"running '{abbreviate(shell.last_command or '?')}' for {shell.get_pending_for()}"


def full_roster(chat: ChatWorkspace) -> str:
    """Every session this chat has, idle ones included. What list_sessions is for."""
    lines = [
        f"  {name}: {_activity(sh, name)} (cwd={sh.cwd})"
        for name, sh in [("main", chat.main), *chat.sessions.items()]
    ]
    return "\n".join([f"[chat {chat.chat_id}] sessions:", *lines])


def roster_footer(chat: ChatWorkspace, exclude: str) -> str:
    """What the *other* sessions are doing — empty when that is nothing.

    Every shell-tool reply used to end in the full roster, including the session
    the call was already about and including the idle ones. So an agent waiting on
    a slow build got the same few hundred tokens back every five seconds: the
    polled session's whole command in the status line, then that same command a
    second time in the roster, then a list of shells sitting idle. Nothing in it
    had changed since the previous poll, and none of it was what was asked.

    A reply already says what the caller's own session is doing. What it cannot
    say is that some *other* session has finished or has stopped to ask a
    question — and that is the only thing a footer is for. Idle is not news, and
    repeating the question back is not an answer.

    full_roster is still there for list_sessions, which exists to be asked.
    """
    lines = [
        f"  {name}: {_activity(sh, name)}"
        for name, sh in [("main", chat.main), *chat.sessions.items()]
        if name != exclude and sh.state == "pending"
    ]
    return "\n".join(["other sessions:", *lines]) if lines else ""


async def warmup_shell(shell: BashState) -> None:
    """Absorb a freshly spawned shell's startup banner.

    A new pty emits a prompt before it is ready; the first real command would
    otherwise match against that banner and come back empty. One throwaway
    command consumes it. Every chat spawns its own shell, so without this every
    chat's first command would look like it did nothing. Best-effort — a failure
    here just means the banner wasn't there to begin with.
    """
    try:
        await anyio.to_thread.run_sync(execute_bash, shell, Command(command="true"), None, None)
    except (pexpect.ExceptionPexpect, OSError) as e:
        # The shell is unusable, not merely un-warmed — but say so rather than
        # letting the first real command fail with no explanation.
        shell.console.log(f"Could not warm up shell {shell.shell_id}: {e}")
