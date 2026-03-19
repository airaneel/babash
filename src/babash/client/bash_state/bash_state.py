import datetime
import os
import re
import threading
import traceback
import time
from hashlib import sha256
from typing import (
    Any,
    Literal,
    NamedTuple,
    Optional,
)
from uuid import uuid4

import pexpect  # type: ignore[import-untyped]

from ...types_ import (
    Console,
    Modes,
)
from ..modes import BashCommandMode, FileEditMode, WriteIfEmptyMode
from .file_whitelist import FileWhitelistData
from .persistence import (
    generate_thread_id,
    load_bash_state_by_id,
    save_bash_state_by_id,
)
from .shell_process import (
    CONFIG,
    PROMPT_CONST,
    check_if_screen_command_available,
    cleanup_orphaned_babash_screens,
    ensure_babash_block_in_rc_file,
    get_rc_file_path,
    get_tmpdir,
    start_shell,
)

BASH_CLF_OUTPUT = Literal["repl", "pending"]


class BashStateSnapshot(NamedTuple):
    bash_command_mode: BashCommandMode
    file_edit_mode: FileEditMode
    write_if_empty_mode: WriteIfEmptyMode
    mode: Modes
    whitelist_for_overwrite: dict[str, Any]  # dict[str, FileWhitelistData] at runtime
    workspace_root: str
    thread_id: str


class BashState:
    _use_screen: bool
    _current_thread_id: str

    def __init__(
        self,
        console: Console,
        working_dir: str,
        bash_command_mode: Optional[BashCommandMode],
        file_edit_mode: Optional[FileEditMode],
        write_if_empty_mode: Optional[WriteIfEmptyMode],
        mode: Optional[Modes],
        use_screen: bool,
        whitelist_for_overwrite: Optional[dict[str, "FileWhitelistData"]],
        thread_id: Optional[str],
        shell_path: Optional[str],
    ) -> None:
        self.last_command: str = ""
        self.console = console
        self._cwd = working_dir or os.getcwd()
        self._workspace_root = working_dir or os.getcwd()
        self._bash_command_mode: BashCommandMode = bash_command_mode or BashCommandMode(
            "normal_mode", "all"
        )
        self._file_edit_mode: FileEditMode = file_edit_mode or FileEditMode("all")
        self._write_if_empty_mode: WriteIfEmptyMode = (
            write_if_empty_mode or WriteIfEmptyMode("all")
        )
        self._mode: Modes = mode or "babash"
        self._whitelist_for_overwrite: dict[str, FileWhitelistData] = (
            whitelist_for_overwrite or {}
        )
        self._current_thread_id = (
            thread_id if thread_id is not None else generate_thread_id()
        )
        self._bg_expect_thread: Optional[threading.Thread] = None
        self._bg_expect_thread_stop_event = threading.Event()
        self._use_screen = use_screen
        self._shell_path: str = (
            shell_path if shell_path else os.environ.get("SHELL", "/bin/bash")
        )
        if get_rc_file_path(self._shell_path) is None:
            console.log(
                f"Warning: Unsupported shell: {self._shell_path}, defaulting to /bin/bash"
            )
            self._shell_path = "/bin/bash"

        self.background_shells = dict[str, BashState]()
        self._init_shell()

    def start_new_bg_shell(self, working_dir: str) -> "BashState":
        cid = uuid4().hex[:10]
        state = BashState(
            self.console,
            working_dir=working_dir,
            bash_command_mode=self.bash_command_mode,
            file_edit_mode=self.file_edit_mode,
            write_if_empty_mode=self.write_if_empty_mode,
            mode=self.mode,
            use_screen=self.over_screen,
            whitelist_for_overwrite=None,
            thread_id=cid,
            shell_path=self._shell_path,
        )
        self.background_shells[cid] = state
        return state

    def expect(
        self, pattern: Any, timeout: Optional[float] = -1, flush_rem_prompt: bool = True
    ) -> int:
        self.close_bg_expect_thread()
        try:
            output = self._shell.expect(pattern, timeout)
            if isinstance(self._shell.match, re.Match) and self._shell.match.groups():
                cwd = self._shell.match.group(1)
                if cwd.strip():
                    self._cwd = cwd
                    if flush_rem_prompt:
                        temp_before = self._shell.before
                        self.flush_prompt()
                        self._shell.before = temp_before
        except pexpect.TIMEOUT:
            return 1
        return int(output)

    def flush_prompt(self) -> None:
        for _ in range(200):
            try:
                output = self.expect([" ", pexpect.TIMEOUT], 0.1)
                if output == 1:
                    return
            except pexpect.TIMEOUT:
                return

    def send(self, s: str | bytes, set_as_command: Optional[str]) -> int:
        if set_as_command is not None:
            self.last_command = set_as_command
        output = self._shell.send(s)
        return int(output)

    def sendline(self, s: str | bytes, set_as_command: Optional[str]) -> int:
        if set_as_command is not None:
            self.last_command = set_as_command
        output = self._shell.sendline(s)
        return int(output)

    @property
    def linesep(self) -> str:
        return str(self._shell.linesep)

    def sendintr(self) -> None:
        self.close_bg_expect_thread()
        self._shell.sendintr()

    @property
    def before(self) -> Optional[str]:
        before = self._shell.before
        if before and before.startswith(self.last_command):
            return str(before[len(self.last_command) :])
        return str(before) if before else None

    def run_bg_expect_thread(self) -> None:
        """Run background expect thread for handling shell interactions."""

        def _bg_expect_thread_handler() -> None:
            while True:
                if self._bg_expect_thread_stop_event.is_set():
                    break
                output = self._shell.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=0.1)
                if output == 0:
                    break

        if self._bg_expect_thread:
            self.close_bg_expect_thread()

        self._bg_expect_thread = threading.Thread(
            target=_bg_expect_thread_handler,
        )
        self._bg_expect_thread.start()
        for _k, v in self.background_shells.items():
            v.run_bg_expect_thread()

    def close_bg_expect_thread(self) -> None:
        if self._bg_expect_thread:
            self._bg_expect_thread_stop_event.set()
            self._bg_expect_thread.join()
            self._bg_expect_thread = None
            self._bg_expect_thread_stop_event = threading.Event()
        for _k, v in self.background_shells.items():
            v.close_bg_expect_thread()

    def cleanup(self) -> None:
        self.close_bg_expect_thread()
        self._shell.close(True)

    def __enter__(self) -> "BashState":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, tb: Any) -> None:
        self.cleanup()

    @property
    def mode(self) -> Modes:
        return self._mode

    @property
    def bash_command_mode(self) -> BashCommandMode:
        return self._bash_command_mode

    @property
    def file_edit_mode(self) -> FileEditMode:
        return self._file_edit_mode

    @property
    def write_if_empty_mode(self) -> WriteIfEmptyMode:
        return self._write_if_empty_mode

    def _init_shell(self) -> None:
        self._state: Literal["repl"] | datetime.datetime = "repl"
        self.last_command = ""
        os.makedirs(self._cwd, exist_ok=True)

        ensure_babash_block_in_rc_file(self._shell_path, self.console)

        if check_if_screen_command_available():
            cleanup_orphaned_babash_screens(self.console)

        self.__shell: pexpect.spawn[str] | None = None
        self.__shell_id: str | None = None
        self._over_screen: bool | None = None

        def _start() -> None:
            try:
                self.__shell, self.__shell_id = start_shell(
                    self._bash_command_mode.bash_mode == "restricted_mode",
                    self._cwd,
                    self.console,
                    over_screen=self._use_screen,
                    shell_path=self._shell_path,
                )
                self._over_screen = self._use_screen
            except Exception as e:
                if not isinstance(e, ValueError):
                    self.console.log(traceback.format_exc())
                self.console.log("Retrying without using screen")
                self.__shell, self.__shell_id = start_shell(
                    self._bash_command_mode.bash_mode == "restricted_mode",
                    self._cwd,
                    self.console,
                    over_screen=False,
                    shell_path=self._shell_path,
                )
                self._over_screen = False

        self._init_thread = threading.Thread(target=_start)
        self._init_thread.start()

        self._pending_output = ""

        self.run_bg_expect_thread()

    @property
    def _shell(self) -> "pexpect.spawn[str]":
        if self.__shell is None:
            self._init_thread.join()
            assert self.__shell
        return self.__shell

    @property
    def _shell_id(self) -> str:
        if self.__shell_id is None:
            self._init_thread.join()
            assert self.__shell_id
        return self.__shell_id

    @property
    def over_screen(self) -> bool:
        if self._over_screen is None:
            self._init_thread.join()
            assert self._over_screen is not None
            return self._over_screen
        return self._over_screen

    def set_pending(self, last_pending_output: str) -> None:
        if not isinstance(self._state, datetime.datetime):
            self._state = datetime.datetime.now()
        self._pending_output = last_pending_output

    def set_repl(self) -> None:
        self._state = "repl"
        self._pending_output = ""
        self.last_command = ""

    def clear_to_run(self) -> None:
        """Check if prompt is clear to enter new command otherwise send ctrl c"""
        starttime = time.time()
        self.close_bg_expect_thread()
        try:
            while True:
                try:
                    output = self.expect(
                        [PROMPT_CONST, pexpect.TIMEOUT], 0.1, flush_rem_prompt=False
                    )
                    if output == 1:
                        break
                except pexpect.TIMEOUT:
                    break
                if time.time() - starttime > CONFIG.timeout:
                    self.console.log(
                        f"Error: could not clear output in {CONFIG.timeout} seconds. Resetting"
                    )
                    self.reset_shell()
                    return
            output = self.expect([" ", pexpect.TIMEOUT], 0.1)
            if output != 1:
                self.send("\x03", None)

                output = self.expect([PROMPT_CONST, pexpect.TIMEOUT], CONFIG.timeout)
                if output == 1:
                    self.console.log("Error: could not clear output. Resetting")
                    self.reset_shell()
        finally:
            self.run_bg_expect_thread()

    @property
    def state(self) -> BASH_CLF_OUTPUT:
        if self._state == "repl":
            return "repl"
        return "pending"

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def workspace_root(self) -> str:
        return self._workspace_root

    def set_workspace_root(self, workspace_root: str) -> None:
        self._workspace_root = workspace_root

    @property
    def prompt(self) -> re.Pattern[str]:
        return PROMPT_CONST

    def reset_shell(self) -> None:
        self.cleanup()
        self._init_shell()

    @property
    def current_thread_id(self) -> str:
        return self._current_thread_id

    def load_state_from_thread_id(self, thread_id: str) -> bool:
        """Load bash state from a thread_id."""
        loaded_state = load_bash_state_by_id(thread_id)
        if not loaded_state:
            return False

        snapshot = BashState.parse_state(loaded_state)
        self.load_state(
            snapshot.bash_command_mode,
            snapshot.file_edit_mode,
            snapshot.write_if_empty_mode,
            snapshot.mode,
            snapshot.whitelist_for_overwrite,
            snapshot.workspace_root,
            snapshot.workspace_root,
            thread_id,
        )
        return True

    def serialize(self) -> dict[str, Any]:
        """Serialize BashState to a dictionary for saving"""
        return {
            "bash_command_mode": self._bash_command_mode.serialize(),
            "file_edit_mode": self._file_edit_mode.serialize(),
            "write_if_empty_mode": self._write_if_empty_mode.serialize(),
            "whitelist_for_overwrite": {
                k: v.serialize() for k, v in self._whitelist_for_overwrite.items()
            },
            "mode": self._mode,
            "workspace_root": self._workspace_root,
            "chat_id": self._current_thread_id,
        }

    def save_state_to_disk(self) -> None:
        """Save the current bash state to disk using the thread_id."""
        state_dict = self.serialize()
        save_bash_state_by_id(self._current_thread_id, state_dict)

    @staticmethod
    def parse_state(
        state: dict[str, Any],
    ) -> BashStateSnapshot:
        whitelist_state = state["whitelist_for_overwrite"]
        whitelist_dict: dict[str, Any] = {}
        if isinstance(whitelist_state, dict):
            for file_path, data in whitelist_state.items():
                if isinstance(data, dict) and "file_hash" in data:
                    whitelist_dict[file_path] = FileWhitelistData.deserialize(data)
                else:
                    whitelist_dict[file_path] = FileWhitelistData(
                        file_hash=data if isinstance(data, str) else "",
                        line_ranges_read=[(1, 1000000)],
                        total_lines=1000000,
                    )
        else:
            whitelist_dict = {
                k: FileWhitelistData(
                    file_hash="", line_ranges_read=[(1, 1000000)], total_lines=1000000
                )
                for k in whitelist_state
            }

        thread_id = state.get("chat_id")
        if thread_id is None:
            thread_id = generate_thread_id()

        return BashStateSnapshot(
            bash_command_mode=BashCommandMode.deserialize(state["bash_command_mode"]),
            file_edit_mode=FileEditMode.deserialize(state["file_edit_mode"]),
            write_if_empty_mode=WriteIfEmptyMode.deserialize(state["write_if_empty_mode"]),
            mode=state["mode"],
            whitelist_for_overwrite=whitelist_dict,
            workspace_root=state.get("workspace_root", ""),
            thread_id=thread_id,
        )

    def load_state(
        self,
        bash_command_mode: BashCommandMode,
        file_edit_mode: FileEditMode,
        write_if_empty_mode: WriteIfEmptyMode,
        mode: Modes,
        whitelist_for_overwrite: dict[str, "FileWhitelistData"],
        cwd: str,
        workspace_root: str,
        thread_id: str,
    ) -> None:
        """Load state into this BashState instance."""
        self._bash_command_mode = bash_command_mode
        self._cwd = cwd or self._cwd
        self._workspace_root = workspace_root or cwd or self._workspace_root
        self._file_edit_mode = file_edit_mode
        self._write_if_empty_mode = write_if_empty_mode
        self._whitelist_for_overwrite = dict(whitelist_for_overwrite)
        self._mode = mode
        self._current_thread_id = thread_id
        self.reset_shell()

        self.save_state_to_disk()

    def get_pending_for(self) -> str:
        if isinstance(self._state, datetime.datetime):
            timedelta = datetime.datetime.now() - self._state
            return (
                str(
                    int(
                        (
                            timedelta + datetime.timedelta(seconds=CONFIG.timeout)
                        ).total_seconds()
                    )
                )
                + " seconds"
            )

        return "Not pending"

    @property
    def whitelist_for_overwrite(self) -> dict[str, "FileWhitelistData"]:
        return self._whitelist_for_overwrite

    def add_to_whitelist_for_overwrite(
        self, file_paths_with_ranges: dict[str, list[tuple[int, int]]]
    ) -> None:
        """Add files to the whitelist for overwrite."""
        for file_path, ranges in file_paths_with_ranges.items():
            with open(file_path, "rb") as f:
                file_content = f.read()
                file_hash = sha256(file_content).hexdigest()
                total_lines = file_content.count(b"\n") + 1

            if file_path in self._whitelist_for_overwrite:
                whitelist_data = self._whitelist_for_overwrite[file_path]
                whitelist_data.file_hash = file_hash
                whitelist_data.total_lines = total_lines
                for range_start, range_end in ranges:
                    whitelist_data.add_range(range_start, range_end)
            else:
                self._whitelist_for_overwrite[file_path] = FileWhitelistData(
                    file_hash=file_hash,
                    line_ranges_read=list(ranges),
                    total_lines=total_lines,
                )

        self.save_state_to_disk()

    @property
    def pending_output(self) -> str:
        return self._pending_output
