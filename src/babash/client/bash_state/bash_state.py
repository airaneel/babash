"""One interactive shell, kept alive across tool calls."""

import datetime
import os
import re
import threading
import time
from typing import Any, Literal, Optional
from uuid import uuid4

import pexpect

from ...settings import ShellTimings
from ...types_ import Console
from .shell_process import (
    PROMPT_CONST,
    TerminalRenderer,
    check_if_screen_command_available,
    cleanup_orphaned_babash_screens,
    ensure_babash_block_in_rc_file,
    get_rc_file_path,
    start_shell,
)

ShellState = Literal["repl", "pending"]

# A bound on how many trailing bytes we will swallow after a prompt. It only has
# to cover the ` \r\e[2K` PROMPT_COMMAND prints; the cap is a backstop against
# spinning forever if the pty is producing garbage.
_MAX_PROMPT_TAIL_READS = 200


def new_shell_id() -> str:
    return f"i{uuid4().hex[:12]}"


class BashState:
    """A pty running a shell, plus what we know about what it's doing.

    The only state here is what genuinely cannot be derived: the pty itself, the
    renderer's position in its output stream, whether a command is outstanding
    and since when, and what the last one exited with. Everything else — the
    cwd, the exit code — is read back out of the shell's own prompt rather than
    tracked in parallel with it.
    """

    _last_exit_code: Optional[int]

    def __init__(
        self,
        console: Console,
        working_dir: str,
        use_screen: bool,
        shell_id: Optional[str],
        shell_path: Optional[str],
        timings: ShellTimings,
    ) -> None:
        self.last_command: str = ""
        self.console = console
        self.timings = timings
        self._cwd = working_dir or os.getcwd()
        self._shell_id = shell_id if shell_id is not None else new_shell_id()
        self._idle_reader: Optional[threading.Thread] = None
        self._idle_reader_stop = threading.Event()
        self._use_screen = use_screen
        self._shell_path: str = shell_path or os.environ.get("SHELL", "/bin/bash")
        if get_rc_file_path(self._shell_path) is None:
            console.log(
                f"Warning: Unsupported shell: {self._shell_path}, defaulting to /bin/bash"
            )
            self._shell_path = "/bin/bash"

        self._init_shell()

    # --- lifecycle ---

    def _init_shell(self) -> None:
        self._state: Literal["repl"] | datetime.datetime = "repl"
        self.last_command = ""
        os.makedirs(self._cwd, exist_ok=True)

        ensure_babash_block_in_rc_file(self._shell_path, self.console)

        if check_if_screen_command_available():
            cleanup_orphaned_babash_screens(self.console)

        self.__shell: pexpect.spawn[str] | None = None
        self.__screen_name: str | None = None

        def _start() -> None:
            # start_shell decides for itself whether screen is usable and reports
            # back by naming the session it made, or not naming one. There is no
            # failure to catch here: asking for screen where there is none is a
            # preference that goes unmet, not an error.
            self.__shell, self.__screen_name = start_shell(
                self._cwd,
                self.console,
                over_screen=self._use_screen,
                shell_path=self._shell_path,
                unique_id=self._shell_id,
            )

        # Spawning a pty is slow enough to be worth overlapping with whatever the
        # caller does next; `_shell` blocks on this thread the moment anyone
        # actually needs the shell.
        self._init_thread = threading.Thread(target=_start)
        self._init_thread.start()

        self._renderer = TerminalRenderer()
        self._last_exit_code = None
        self._awaiting_input: str | None = None

        self.start_idle_reader()

    def reset_shell(self) -> None:
        self.cleanup()
        self._init_shell()

    def cleanup(self) -> None:
        self.stop_idle_reader()
        self._shell.close(True)

    def __enter__(self) -> "BashState":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, tb: Any) -> None:
        self.cleanup()

    @property
    def _shell(self) -> "pexpect.spawn[str]":
        if self.__shell is None:
            self._init_thread.join()
            assert self.__shell
        return self.__shell

    @property
    def over_screen(self) -> bool:
        """Whether this shell ended up wrapped in a screen session. Derived from
        whether one got named, rather than tracked alongside it."""
        self._init_thread.join()
        return self.__screen_name is not None

    @property
    def shell_id(self) -> str:
        return self._shell_id

    # --- talking to the pty ---

    def expect(
        self, pattern: Any, timeout: Optional[float] = -1, flush_rem_prompt: bool = True
    ) -> int:
        """Wait for one of `pattern`, and return the index of what matched.

        This never raises on a timeout — it returns 1, the index callers give
        `pexpect.TIMEOUT` in their pattern list. Which is why nothing that calls
        it needs to guard against TIMEOUT, and nothing does.
        """
        self.stop_idle_reader()
        try:
            index = int(self._shell.expect(pattern, timeout))
        except pexpect.TIMEOUT:
            return 1

        match = self._shell.match
        # Only a prompt match has groups (see PROMPT_CONST); everything else is a
        # plain pattern with nothing to say.
        if isinstance(match, re.Match) and match.groups():
            self._read_prompt(match, flush_rem_prompt)
        return index

    def _read_prompt(self, match: re.Match[str], flush_tail: bool) -> None:
        """Take the finished command's exit code and cwd out of the prompt."""
        self._last_exit_code = int(match.group(1))
        cwd = match.group(2)
        if not cwd.strip():
            return
        self._cwd = cwd
        if not flush_tail:
            return

        # Draining reads more from the pty, and pexpect overwrites `before` when
        # it does — but `before` is the command output the caller is waiting for.
        # Put it back.
        before = self._shell.before
        self._drain_prompt_tail()
        self._shell.before = before

    def _drain_prompt_tail(self) -> None:
        """Swallow the bytes the prompt prints after its sentinel.

        PROMPT_COMMAND emits ` \\r\\e[2K` once it has written `──➤`, to wipe the
        line. Left in the buffer, that trailing run would be read back as the
        start of the next command's output.
        """
        for _ in range(_MAX_PROMPT_TAIL_READS):
            if self.expect([" ", pexpect.TIMEOUT], 0.1) == 1:
                return

    def send(self, s: str | bytes, set_as_command: Optional[str]) -> int:
        if set_as_command is not None:
            self.last_command = set_as_command
        return int(self._shell.send(s))

    def sendintr(self) -> None:
        self.stop_idle_reader()
        self._shell.sendintr()

    @property
    def linesep(self) -> str:
        return str(self._shell.linesep)

    @property
    def before(self) -> Optional[str]:
        before = self._shell.before
        if before and before.startswith(self.last_command):
            return str(before[len(self.last_command) :])
        return str(before) if before else None

    @property
    def prompt(self) -> re.Pattern[str]:
        return PROMPT_CONST

    def clear_to_run(self) -> None:
        """Get the pty to a clean prompt before a new command goes in.

        Whatever a previous command left unread would otherwise come back as the
        new one's output. Two things can be in the way: leftover prompts, which
        are simply consumed, and a program still writing, which has to be
        interrupted. If neither clears, the shell is wedged and gets restarted —
        losing its cwd and env is bad, but far less bad than answering the next
        command with the last one's output.
        """
        deadline = time.time() + self.timings.command_budget
        self.stop_idle_reader()
        try:
            # expect() returns 0 for a prompt and 1 for "nothing more came" —
            # so this drains prompts until the buffer goes quiet.
            while self.expect([PROMPT_CONST, pexpect.TIMEOUT], 0.1, flush_rem_prompt=False) == 0:
                if time.time() > deadline:
                    self.console.log(
                        f"Could not clear output in {self.timings.command_budget}s. Resetting."
                    )
                    self.reset_shell()
                    return

            # Still bytes arriving with no prompt behind them: something is
            # running. Interrupt it and wait for the prompt that follows.
            if self.expect([" ", pexpect.TIMEOUT], 0.1) != 1:
                self.send("\x03", None)
                if self.expect([PROMPT_CONST, pexpect.TIMEOUT], self.timings.command_budget) == 1:
                    self.console.log("Could not clear output after Ctrl-c. Resetting.")
                    self.reset_shell()
        finally:
            self.start_idle_reader()

    # --- the idle reader ---
    #
    # A pty has a finite kernel buffer. A command that keeps printing while no
    # tool call is reading will fill it and then block — the command stops making
    # progress, not because it is slow, but because nobody is listening. Between
    # calls, this thread is the listener.
    #
    # It doesn't consume anything: pexpect keeps what it read in the buffer when
    # a wait times out (`spawn.before = spawn._before.getvalue()`), so the next
    # real expect() still sees every byte. It only keeps the pipe flowing.

    def _drain_idle_pty(self) -> None:
        while not self._idle_reader_stop.is_set():
            if self._shell.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=0.1) == 0:
                return  # the shell is gone; nothing left to drain

    def start_idle_reader(self) -> None:
        """Start the idle reader. Safe to call when one is already running."""
        self.stop_idle_reader()
        self._idle_reader = threading.Thread(target=self._drain_idle_pty, daemon=True)
        self._idle_reader.start()

    def stop_idle_reader(self) -> None:
        """Stop the idle reader, so a tool call can have the pty to itself."""
        if self._idle_reader is None:
            return
        self._idle_reader_stop.set()
        self._idle_reader.join()
        self._idle_reader = None
        self._idle_reader_stop = threading.Event()

    # --- what the shell is doing ---

    def incremental_output(self) -> str:
        """Rendered output that has appeared since the last time this was asked.

        The renderer, not this call, holds the "what have we already shown"
        state — so polling a long-running command costs only the new bytes.
        """
        return self._renderer.incremental(self.before or "")

    def cursor_prompt(self) -> str | None:
        """Raw signal: the partial line the cursor is parked in, if any."""
        return self._renderer.cursor_prompt()

    def set_awaiting_input(self, prompt: str | None) -> None:
        """Record that the shell is blocked on `prompt` (or no longer is)."""
        self._awaiting_input = prompt

    def pending_prompt(self) -> str | None:
        """The prompt this shell is blocked on, if it is blocked on one.

        `state == "pending"` alone cannot tell an agent whether a command is
        slow or is sitting waiting to be answered — both simply stop producing
        output, and an agent that mistakes the second for the first will poll a
        command that can never move until it is answered. execute_bash works out
        which it is; this is where the answer is kept, so the session roster can
        say so too.
        """
        return self._awaiting_input if self.state == "pending" else None

    def set_pending(self) -> None:
        if not isinstance(self._state, datetime.datetime):
            self._state = datetime.datetime.now()

    def set_repl(self) -> None:
        self._state = "repl"
        self.last_command = ""
        self._awaiting_input = None
        # The command is done; the next one starts on a clean screen.
        self._renderer.reset()

    @property
    def state(self) -> ShellState:
        return "repl" if self._state == "repl" else "pending"

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def last_exit_code(self) -> Optional[int]:
        """Exit code of the last command to reach a prompt, or None if none has."""
        return self._last_exit_code

    def get_pending_for(self) -> str:
        if not isinstance(self._state, datetime.datetime):
            return "Not pending"
        elapsed = datetime.datetime.now() - self._state
        seconds = int(
            (elapsed + datetime.timedelta(seconds=self.timings.command_budget)).total_seconds()
        )
        return f"{seconds} seconds"
