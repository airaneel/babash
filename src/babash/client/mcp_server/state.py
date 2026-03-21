"""Server state types and session management."""

import logging
from dataclasses import dataclass
from typing import Any

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
class AppState:
    bash_state: BashState
    custom_instructions: str | None
    console: Console
    initialized: bool = False
    sessions: dict[str, BashState] | None = None
    last_output: dict[str, str] | None = None
    history: list[CommandRecord] | None = None

    def get_sessions(self) -> dict[str, BashState]:
        if self.sessions is None:
            self.sessions = {}
        return self.sessions

    def get_last_outputs(self) -> dict[str, str]:
        if self.last_output is None:
            self.last_output = {}
        return self.last_output

    def get_history(self) -> list[CommandRecord]:
        if self.history is None:
            self.history = []
        return self.history

    def get_shell(self, session: str | None) -> BashState:
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
