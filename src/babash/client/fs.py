"""Reading and writing files — here, or wherever a session's shell happens to be.

A session may be sitting inside an SSH connection, in which case the file lives
on the far end of it and the only way to reach it is to send commands down the
same pty. `FileStore` is the seam: the tools are written once, against this, and
neither knows nor cares which side of a network the file is on.

The remote side moves content as base64 rather than as a heredoc. A heredoc has
to survive the remote shell's parser, and anything containing quotes, `$`, or
backticks does not — which is exactly the YAML, Jinja, and JSON an agent most
often wants to write. base64 has no metacharacters, so nothing can be
misinterpreted on the way.
"""

import base64
import binascii
import os
import shlex
from typing import Protocol

from ..types_ import Command
from .bash_state import BashState, execute_bash

# Enough for any single command we issue to read a file back.
_READ_MAX_CHARS = 400_000


class FileError(Exception):
    """The store could not do it, and the agent needs to be told why."""


class FileStore(Protocol):
    """Somewhere files live."""

    def read(self, path: str) -> str:
        """The file's contents as text. Raises FileError if it isn't there."""
        ...

    def read_bytes(self, path: str) -> bytes:
        """The file's raw bytes — for anything that isn't text, like an image."""
        ...

    def write(self, path: str, content: str) -> None:
        """Create or overwrite the file, making parent directories as needed."""
        ...

    def exists(self, path: str) -> bool: ...

    @property
    def where(self) -> str:
        """Human-readable location, for messages: 'locally' or "in session 'x'"."""
        ...


class LocalStore:
    """Files on the machine babash itself runs on."""

    @property
    def where(self) -> str:
        return "locally"

    def read(self, path: str) -> str:
        return self.read_bytes(path).decode("utf-8", errors="replace")

    def read_bytes(self, path: str) -> bytes:
        try:
            with open(path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            raise FileError(f"File does not exist: {path}")
        except IsADirectoryError:
            raise FileError(f"Not a file: {path}")
        except OSError as e:
            raise FileError(f"Could not read {path}: {e}")

    def write(self, path: str, content: str) -> None:
        parent = os.path.dirname(path)
        try:
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            raise FileError(f"Could not write {path}: {e}")

    def exists(self, path: str) -> bool:
        return os.path.isfile(path)


class SessionStore:
    """Files wherever a session's shell is — possibly across an SSH connection.

    Every operation is a command sent down the pty, so the session has to be
    idle: if a command is still running, the shell would feed our `cat` to *it*
    rather than to bash.
    """

    def __init__(self, shell: BashState, session_name: str) -> None:
        self._shell = shell
        self._name = session_name

    @property
    def where(self) -> str:
        return f"in session '{self._name}'"

    def _run(self, command: str) -> str:
        if self._shell.state == "pending":
            raise FileError(
                f"Session '{self._name}' is busy running "
                f"'{self._shell.last_command or 'unknown'}'. Use "
                f"check_status(session='{self._name}') or "
                f"send_keys('Ctrl-c', session='{self._name}') first."
            )
        output = execute_bash(
            self._shell, Command(command=command), _READ_MAX_CHARS, None
        )
        body, _, _ = output.partition("\n\n---\n\n")
        return body

    def read(self, path: str) -> str:
        return self.read_bytes(path).decode("utf-8", errors="replace")

    def read_bytes(self, path: str) -> bytes:
        if not self.exists(path):
            raise FileError(f"File does not exist {self.where}: {path}")
        # base64 on the way back too. The file's bytes have to cross a terminal
        # to get here, and a terminal will happily interpret some of them as
        # control sequences — which for a text file mangles it, and for a PNG
        # destroys it.
        encoded = self._run(f"base64 < {shlex.quote(path)}")
        try:
            return base64.b64decode("".join(encoded.split()))
        except (binascii.Error, ValueError) as e:
            raise FileError(f"Could not decode {path} from {self.where}: {e}")

    def write(self, path: str, content: str) -> None:
        encoded = base64.b64encode(content.encode()).decode()
        parent = os.path.dirname(path)
        if parent:
            self._run(f"mkdir -p {shlex.quote(parent)}")
        self._run(f"printf '%s' {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}")

        written = self._run(f"wc -c < {shlex.quote(path)}").strip()
        expected = len(content.encode())
        # Trust but verify: a full disk or a read-only mount fails quietly here
        # in a way it never would locally.
        if written.isdigit() and int(written) != expected:
            raise FileError(
                f"Wrote {path} {self.where} but it is {written} bytes, expected {expected}."
            )

    def exists(self, path: str) -> bool:
        return "EXISTS" in self._run(
            f"test -f {shlex.quote(path)} && echo EXISTS || echo NO"
        )
