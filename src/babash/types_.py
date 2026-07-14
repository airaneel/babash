"""The things you can say to a shell.

These are plain dataclasses, not pydantic models. They used to be models with
discriminated unions, `extra="ignore"`, and a pile of validators whose job was
to repair malformed JSON an LLM had typed by hand — because upstream, the model
really did hand-author the whole payload. It doesn't anymore: FastMCP validates
at the tool boundary from each tool's own signature, and the tool then builds
one of these itself, in typed code. Nothing untrusted reaches them, so there is
nothing left to validate.
"""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

Specials = Literal[
    # Confirm / edit
    "Enter",
    "Tab",
    "Backspace",
    "Escape",
    # Move around a TUI or a pager
    "Key-up",
    "Key-down",
    "Key-left",
    "Key-right",
    "Home",
    "End",
    "PageUp",
    "PageDown",
    # Control the running program
    "Ctrl-c",  # interrupt
    "Ctrl-d",  # end of input — closes a REPL, ends `cat > file`
    "Ctrl-z",  # suspend
    "Ctrl-l",  # redraw a garbled screen
]


@dataclass(frozen=True)
class Command:
    """Run a command in the shell."""

    command: str


@dataclass(frozen=True)
class StatusCheck:
    """Ask a running command what it's done since we last looked."""


@dataclass(frozen=True)
class SendText:
    """Type text into whatever is running (a password, a prompt answer)."""

    send_text: str


@dataclass(frozen=True)
class SendSpecials:
    """Press keys that aren't text — Ctrl-c, arrows, Enter."""

    send_specials: tuple[Specials, ...]


BashAction = Command | StatusCheck | SendText | SendSpecials


class Console(Protocol):
    def print(self, *objects: Any, **kwargs: Any) -> None: ...

    def log(self, *objects: Any, **kwargs: Any) -> None: ...
