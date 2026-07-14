import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from hashlib import md5
from typing import Optional

import pexpect
import psutil
import pyte
import pyte.modes as pyte_modes

from ...types_ import Console

logger = logging.getLogger("babash")

# The prompt is the sentinel. PROMPT_COMMAND runs after every command and prints
# `◉ <exit code>|<cwd>──➤`, then `\r\e[2K` wipes the line so the user never sees
# it. Matching it tells us three things at once: that the command finished, what
# it exited with, and where the shell now is — so none of that has to be tracked
# by hand or scraped out of the command's own output.
#
# `__babash_ec=$?` must be the first thing executed: the `$(pwd)` substitution
# below is itself a command and would otherwise clobber `$?`. The cwd is passed
# as a printf *argument*, not spliced into the format string, so a directory
# containing a `%` can't corrupt the prompt.
PROMPT_CONST = re.compile(r"◉ (\d+)\|([^\n]*)──➤")
PROMPT_COMMAND = "__babash_ec=$?; printf '◉ %s|%s──➤ \r\\e[2K' \"$__babash_ec\" \"$(pwd)\""
PROMPT_STATEMENT = ""


# Two timeouts that have nothing to do with how long a *command* may run, and
# which used to share a value with it — so turning the command budget down for
# responsiveness also gave a shell two seconds to source ~/.bashrc, after which
# it silently fell back to --norc.
SUBPROCESS_TIMEOUT = 5.0
"""For the small `screen`/`which`/`getconf` calls babash shells out to."""

SHELL_STARTUP_TIMEOUT = 15.0
"""For a shell to boot, source its rc file, and print its first prompt. Generous:
an rc file can be slow, and getting this wrong costs the user their shell config."""


def is_mac() -> bool:
    return platform.system() == "Darwin"


def get_tmpdir() -> str:
    current_tmpdir = os.environ.get("TMPDIR", "")
    if current_tmpdir or not is_mac():
        return tempfile.gettempdir()
    return _run(["getconf", "DARWIN_USER_TEMP_DIR"], SUBPROCESS_TIMEOUT).strip() or tempfile.gettempdir()


def check_if_screen_command_available() -> bool:
    """Whether `screen` is installed, ensuring it has a usable .screenrc if so."""
    if shutil.which("screen") is None:
        return False

    screenrc = os.path.join(os.path.expanduser("~"), ".screenrc")
    if not os.path.exists(screenrc):
        try:
            with open(screenrc, "w") as f:
                f.write("defscrollback 10000\ntermcapinfo xterm* ti@:te@\n")
        except OSError as e:
            # A missing .screenrc costs us scrollback, not correctness.
            logger.debug("could not write %s: %s", screenrc, e)
    return True


def _run(args: list[str], timeout: float) -> str:
    """Run a short command and return its output, or "" if it could not run.

    `screen` writes its session list to stderr on some platforms and stdout on
    others, so both are taken. capture_output is not optional: this process
    speaks MCP JSON-RPC over stdout, and one stray line from a subprocess
    corrupts the channel.
    """
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, check=False, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("%s failed: %s", args[0], e)
        return ""
    return result.stdout or result.stderr or ""


def get_babash_screen_sessions() -> list[str]:
    """Every screen session babash has ever started, as `screen -ls` names them."""
    return [
        session
        for session in (line.split()[0] for line in _run(["screen", "-ls"], 0.5).splitlines() if line.split())
        if ".babash." in session
    ]


def _session_pid(session_id: str) -> Optional[int]:
    """The pid `screen` embeds at the front of a session name, if it is one."""
    head = session_id.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _is_orphaned(pid: int) -> bool:
    """Whether the process that owns a screen session is gone.

    Two ways for that to be true, and NoSuchProcess is one of them rather than an
    error to be swallowed: if the owner has exited outright, the session is
    orphaned by definition. If it merely died and left the session behind, screen
    gets reparented to init.
    """
    try:
        return psutil.Process(pid).ppid() == 1
    except psutil.NoSuchProcess:
        return True


def get_orphaned_babash_screens() -> list[str]:
    """babash screen sessions whose owning process is no longer around."""
    pids = ((session, _session_pid(session)) for session in get_babash_screen_sessions())
    return [session for session, pid in pids if pid is not None and _is_orphaned(pid)]


def cleanup_orphaned_babash_screens(console: Console) -> None:
    orphaned = get_orphaned_babash_screens()
    if not orphaned:
        return

    console.log(f"Found {len(orphaned)} orphaned babash screen sessions to clean up")
    for session in orphaned:
        _run(["screen", "-S", session, "-X", "quit"], SUBPROCESS_TIMEOUT)


def get_rc_file_path(shell_path: str) -> Optional[str]:
    """Get the rc file path for the given shell."""
    shell_name = os.path.basename(shell_path)
    home_dir = os.path.expanduser("~")

    if shell_name == "zsh":
        return os.path.join(home_dir, ".zshrc")
    elif shell_name == "bash":
        return os.path.join(home_dir, ".bashrc")
    else:
        return None


MARKER_START = "# --BABASH_ENVIRONMENT_START--"
MARKER_END = "# --BABASH_ENVIRONMENT_END--"

# Same sentinel as PROMPT_COMMAND, requoted for embedding in an rc file: the
# outer quotes are single, so every quote inside has to be double.
_RC_PROMPT_COMMAND = (
    '__babash_ec=$?; printf "◉ %s|%s──➤ \\r\\e[2K" "$__babash_ec" "$(pwd)"'
)


def babash_rc_block(shell_name: str) -> Optional[str]:
    """The block babash keeps in the user's rc file, or None for shells we don't
    know how to configure. Only the interactive shells babash itself spawns pick
    it up — the `IN_BABASH_ENVIRONMENT` guard keeps it out of the user's own."""
    if shell_name == "zsh":
        body = f""" PROMPT_COMMAND='{_RC_PROMPT_COMMAND}'
 prmptcmdbabash() {{ eval "$PROMPT_COMMAND" }}
 add-zsh-hook -d precmd prmptcmdbabash
 precmd_functions+=prmptcmdbabash"""
    elif shell_name == "bash":
        body = f""" PROMPT_COMMAND='{_RC_PROMPT_COMMAND}'"""
    else:
        return None

    return f"""{MARKER_START}
if [ -n "$IN_BABASH_ENVIRONMENT" ]; then
{body}
fi
{MARKER_END}
"""


def ensure_babash_block_in_rc_file(shell_path: str, console: Console) -> None:
    """Install babash's block in the rc file, or rewrite it if it's out of date.

    Rewriting matters: the block pins the prompt sentinel, and a machine that
    installed an older babash already has a block with the older sentinel in it.
    Bailing out on "marker already present" — which is what this used to do —
    would leave that stale prompt in place forever, and the new PROMPT_CONST
    would never match it.
    """
    rc_file_path = get_rc_file_path(shell_path)
    if not rc_file_path:
        return

    babash_block = babash_rc_block(os.path.basename(shell_path))
    if babash_block is None:
        return

    try:
        with open(rc_file_path) as f:
            content = f.read()
    except OSError:
        content = ""

    start = content.find(MARKER_START)
    end = content.find(MARKER_END)
    if start != -1 and end > start:
        existing = content[start : end + len(MARKER_END)]
        if existing.strip() == babash_block.strip():
            return
        updated = content[:start] + babash_block.strip() + content[end + len(MARKER_END) :]
        action = f"Updated babash environment block in {rc_file_path}"
    elif content:
        updated = content + "\n" + babash_block
        action = f"Added babash environment block to {rc_file_path}"
    else:
        updated = babash_block
        action = f"Created {rc_file_path} with babash environment block"

    try:
        with open(rc_file_path, "w") as f:
            f.write(updated)
        console.log(action)
    except OSError as e:
        console.log(f"Failed to update {rc_file_path}: {e}")


def start_shell(
    initial_dir: str,
    console: Console,
    over_screen: bool,
    shell_path: str,
    unique_id: str,
) -> tuple["pexpect.spawn[str]", Optional[str]]:
    """Spawn a shell and return it, with the name of the screen session wrapping
    it — or None if it isn't wrapped in one.

    `over_screen` is a preference, not a demand: if `screen` isn't installed we
    simply don't use it, and say so in the return value. It used to raise
    ValueError for that, which meant the caller had to catch an exception to
    learn a fact it could have been told.
    """
    cmd = shell_path

    overrideenv = {
        **os.environ,
        "PROMPT_COMMAND": PROMPT_COMMAND,
        "TMPDIR": get_tmpdir(),
        "TERM": "xterm-256color",
        "IN_BABASH_ENVIRONMENT": "1",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
    }
    try:
        shell = pexpect.spawn(
            cmd,
            env=overrideenv,
            echo=True,
            encoding="utf-8",
            timeout=SHELL_STARTUP_TIMEOUT,
            cwd=initial_dir,
            codec_errors="backslashreplace",
            dimensions=(500, 160),
        )
        shell.sendline(PROMPT_STATEMENT)
        shell.expect(PROMPT_CONST, timeout=SHELL_STARTUP_TIMEOUT)
    except (pexpect.ExceptionPexpect, OSError) as e:
        # The prompt never arrived, which means the rc file did not install it —
        # it errored, or hung, or the shell isn't one we know how to configure.
        # A bare shell with no rc still gives us a working, promptable terminal;
        # the user just loses their aliases in it.
        console.log(f"Shell did not reach a babash prompt ({e}). Retrying without rc.")

        shell = pexpect.spawn(
            "/bin/bash --noprofile --norc",
            env=overrideenv,
            echo=True,
            encoding="utf-8",
            timeout=SHELL_STARTUP_TIMEOUT,
            codec_errors="backslashreplace",
        )
        shell.sendline(PROMPT_STATEMENT)
        shell.expect(PROMPT_CONST, timeout=SHELL_STARTUP_TIMEOUT)

    initialdir_hash = md5(
        os.path.normpath(os.path.abspath(initial_dir)).encode()
    ).hexdigest()[:5]
    # unique_id (the owning BashState's thread_id) guarantees a distinct screen
    # name per shell. Without it the name is just timestamp(second)+dir, so two
    # shells started in the same directory within the same second collide.
    name = shlex.quote(
        "babash."
        + time.strftime("%d-%Hh%Mm%Ss")
        + f".{initialdir_hash[:3]}."
        + f"{unique_id}."
        + os.path.basename(initial_dir)
    )

    screen_name: Optional[str] = None
    if over_screen and not check_if_screen_command_available():
        console.log("screen is not installed; running the shell directly.")
    elif over_screen:
        # Drain the outer shell's prompts before handing it to screen, or the
        # inner shell's first prompt gets mixed up with them.
        while shell.expect([PROMPT_CONST, pexpect.TIMEOUT], timeout=0.1) == 0:
            pass
        shell.sendline(f"screen -q -S {name} {shell_path}")
        shell.expect(PROMPT_CONST, timeout=SHELL_STARTUP_TIMEOUT)
        screen_name = name

    # Sync to a clean prompt before handing the shell back. A freshly started
    # shell (especially the inner shell screen spawns) briefly emits its own
    # default prompt banner — e.g. `bash-3.2$` on macOS — before PROMPT_COMMAND
    # installs the babash prompt. If that leftover text is still buffered, the
    # very first real command matches it instead of its own output and comes
    # back empty. Send a blank line and drain every prompt (in short slices, so
    # a missed prompt costs 0.3s, not the full startup timeout) until the buffer
    # goes quiet — the blank-line prompt match consumes the banner before it.
    # (Mirrors pexpect.pxssh.sync_original_prompt.)
    shell.sendline("")
    for _ in range(20):
        if shell.expect([PROMPT_CONST, pexpect.TIMEOUT], timeout=0.3) == 1:
            break

    return shell, screen_name


def _incremental_lines(old_output: list[str], new_output: list[str]) -> list[str]:
    """The tail of `new_output` that comes after `old_output`'s content.

    Anchors on the last line of `old_output` and walks backwards to confirm the
    rest lines up, which tolerates the screen having scrolled between renders.
    """
    nold = len(old_output)
    nnew = len(new_output)
    if not old_output:
        return new_output
    for i in range(nnew - 1, -1, -1):
        if new_output[i] != old_output[-1]:
            continue
        for j in range(i - 1, -1, -1):
            if (nold - 1 + j - i) < 0:
                break
            if new_output[j] != old_output[-1 + j - i]:
                break
        else:
            return new_output[i + 1 :]
    return new_output


class TerminalRenderer:
    """Renders one shell's pty stream into screen lines, incrementally.

    pyte's `Stream.feed()` is append-only by design: the `Screen` holds the
    rendered grid and each feed applies only the bytes handed to it. So a
    renderer is only expensive if you throw the Screen away and re-feed the
    whole buffer every time you look at it — which is what babash used to do,
    at ~230ms per poll on a 100KB buffer, to recompute a grid pyte already had.
    Keeping the Screen alive and feeding it only the new bytes makes the cost
    proportional to the *new* output (~20ms, and flat in buffer size).

    That also retires a bug. Re-feeding meant capping the input at the last
    100KB to bound the cost, while the offset used to find "what's new" was
    still measured against the full, uncapped buffer — so once a command had
    printed more than 100KB, the two disagreed, the slice came out empty, and
    every subsequent poll reported no new output at all.

    One renderer belongs to one shell and spans one command: `reset()` when a
    command finishes, so the next one starts from a clean screen.
    """

    def __init__(self) -> None:
        self._screen = pyte.Screen(160, 500)
        self._screen.set_mode(pyte_modes.LNM)
        self._stream = pyte.Stream(self._screen)
        self._fed = 0
        self._reported: list[str] = []

    def reset(self) -> None:
        self._screen.reset()
        self._screen.set_mode(pyte_modes.LNM)
        self._fed = 0
        self._reported = []

    def _display(self) -> list[str]:
        """The screen with its trailing blank lines dropped."""
        dsp = self._screen.display[::-1]
        for i, line in enumerate(dsp):
            if line.strip():
                break
        else:
            i = len(dsp)
        return self._screen.display[: len(dsp) - i]

    def cursor_prompt(self) -> str | None:
        """The partial line the cursor is sitting in, if it's sitting in one.

        A program that has written `Select [1/2/3]: ` and is now blocked on read()
        leaves the cursor parked partway along a line, with no newline after it.
        That is what a prompt awaiting input looks like on a screen — and it is
        the only thing that distinguishes it from a command that is merely slow,
        since both simply stop producing output.

        This is only the raw signal, not the verdict: a progress bar redrawing
        itself also parks the cursor mid-line. `execute_bash` confirms it by
        watching for a beat and seeing whether anything more arrives.
        """
        if self._screen.cursor.x == 0:
            return None
        line = self._screen.display[self._screen.cursor.y][: self._screen.cursor.x]
        return line.strip() or None

    def incremental(self, buffer: str) -> str:
        """Output that has appeared since the last call.

        `buffer` is everything the shell has emitted for the current command;
        only its unseen tail is fed to pyte. A buffer shorter than what we've
        already consumed means the pty buffer was rewound under us, so the
        screen is rebuilt rather than fed a nonsensical delta.
        """
        if len(buffer) < self._fed:
            self.reset()
        self._stream.feed(buffer[self._fed :])
        self._fed = len(buffer)

        lines = self._display()
        previous = self._reported
        self._reported = lines

        if not previous:
            return _rstrip(lines).lstrip()

        # Anchor on all but the last previously-reported line: the last one may
        # have been incomplete (a progress bar mid-redraw) and grown since, in
        # which case it has to be re-emitted rather than treated as already seen.
        new_lines = _incremental_lines(previous[:-1], lines)
        if new_lines and new_lines[0] == previous[-1]:
            new_lines = new_lines[1:]
        return _rstrip(new_lines)


def _rstrip(lines: list[str]) -> str:
    return "\n".join(line.rstrip() for line in lines)
