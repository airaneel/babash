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


def roster_footer(chat: ChatWorkspace) -> str:
    """Full session roster for this chat, appended to shell-tool responses so the
    agent always sees which shells it has open and which are busy."""

    def fmt(name: str, sh: BashState) -> str:
        if sh.state != "pending":
            return f"  {name}: idle (cwd={sh.cwd})"
        prompt = sh.pending_prompt()
        if prompt:
            return (
                f"  {name}: WAITING FOR INPUT at {prompt!r} — "
                f"send_input(text=..., session={name!r}) (cwd={sh.cwd})"
            )
        return (
            f"  {name}: running '{sh.last_command or '?'}' "
            f"for {sh.get_pending_for()} (cwd={sh.cwd})"
        )

    lines = [f"[chat {chat.chat_id}] sessions:", fmt("main", chat.main)]
    for name, sh in chat.sessions.items():
        lines.append(fmt(name, sh))
    return "\n".join(lines)


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
