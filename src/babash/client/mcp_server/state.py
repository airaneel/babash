"""Server state types and per-chat session management.

Isolation model
---------------
A single babash server process is shared by every conversation that connects to
it (over stdio all of Claude Desktop's chats share one process; over
streamable-http the client does not send a per-conversation id either — see
anthropics/claude-code#41836). The transport therefore gives the server no way
to tell chats apart.

The only signal that *is* per-conversation is a value the model itself carries
in its context and passes on each call: `chat_id`. So all shell state is keyed
by `chat_id`. Each chat gets its own `ChatWorkspace` (an independent "main"
shell plus its own named sessions), which means two chats never share a pty,
cwd, env, or command — no locks required, because within one chat the model
issues calls sequentially.
"""

import logging
from dataclasses import dataclass
from typing import Any

import pexpect

from ...settings import Settings
from ..bash_state import BashState

logger = logging.getLogger("babash")


class Console:
    """Console adapter mapping print/log to Python logging levels."""

    def print(self, msg: str, *args: Any, **kwargs: Any) -> None:
        logger.info(msg)

    def log(self, msg: str, *args: Any, **kwargs: Any) -> None:
        logger.debug(msg)


@dataclass
class CommandRecord:
    command: str
    output: str
    session: str
    success: bool
    errors: list[str]


@dataclass
class ChatWorkspace:
    """All shell state owned by a single chat_id."""

    chat_id: str
    main: BashState
    sessions: dict[str, BashState]
    last_output: dict[str, str]
    history: list[CommandRecord]

    def get_shell(self, session: str | None) -> BashState:
        if not session or session == "main":
            return self.main
        if session not in self.sessions:
            raise ValueError(
                f"Session '{session}' not found in chat '{self.chat_id}'. "
                f"Available: main, {', '.join(self.sessions.keys()) or '(none)'}. "
                f"Create one with create_session."
            )
        return self.sessions[session]

    def all_shells(self) -> list[BashState]:
        return [self.main, *self.sessions.values()]

    def cleanup(self) -> None:
        """Kill every shell this chat owns.

        Nothing here may raise: a pty that fails to close must not stop the next
        one from being closed, and a leaked pty outlives the server.
        """
        for shell in self.all_shells():
            try:
                shell.cleanup()
            except (pexpect.ExceptionPexpect, OSError) as e:
                logger.warning("could not close shell %s: %s", shell.shell_id, e)


@dataclass
class AppState:
    """Process-wide state: the settings needed to spin up shells, plus the live
    per-chat workspaces. Holds no single ambient shell — every shell belongs to
    a chat."""

    settings: Settings
    console: Console
    chats: dict[str, ChatWorkspace]

    def new_shell(self, working_dir: str) -> BashState:
        """Construct (and start) a fresh shell with a unique id, which is what
        keeps its screen session distinct from every other shell's."""
        return BashState(
            console=self.console,
            working_dir=working_dir,
            use_screen=True,
            shell_id=None,
            shell_path=self.settings.shell_path,
            timings=self.settings.timings,
        )

    def get_chat(self, chat_id: str) -> ChatWorkspace | None:
        return self.chats.get(chat_id)

    def create_chat(self, chat_id: str) -> ChatWorkspace:
        """Create a chat's workspace with a fresh 'main' shell. Overwrites any
        existing workspace for this id after cleaning it up (re-initialize)."""
        existing = self.chats.get(chat_id)
        if existing is not None:
            existing.cleanup()
        chat = ChatWorkspace(
            chat_id=chat_id,
            main=self.new_shell(self.settings.workspace),
            sessions={},
            last_output={},
            history=[],
        )
        self.chats[chat_id] = chat
        return chat

    def cleanup(self) -> None:
        for chat in self.chats.values():
            chat.cleanup()
