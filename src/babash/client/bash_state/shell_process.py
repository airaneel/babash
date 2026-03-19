import os
import platform
import re
import shlex
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass
from hashlib import md5
from typing import Optional

import pexpect  # type: ignore[import-untyped]
import psutil  # type: ignore[import-untyped]
import pyte
import pyte.modes as pyte_modes

from ...types_ import Console

PROMPT_CONST = re.compile(r"◉ ([^\n]*)──➤")
PROMPT_COMMAND = "printf '◉ '\"$(pwd)\"'──➤'' \r\\e[2K'"
PROMPT_STATEMENT = ""


@dataclass
class Config:
    timeout: float = 5
    timeout_while_output: float = 20
    output_wait_patience: float = 3

    def update(
        self, timeout: float, timeout_while_output: float, output_wait_patience: float
    ) -> None:
        self.timeout = timeout
        self.timeout_while_output = timeout_while_output
        self.output_wait_patience = output_wait_patience


CONFIG = Config()


def is_mac() -> bool:
    return platform.system() == "Darwin"


def get_tmpdir() -> str:
    current_tmpdir = os.environ.get("TMPDIR", "")
    if current_tmpdir or not is_mac():
        return tempfile.gettempdir()
    try:
        result = subprocess.check_output(
            ["getconf", "DARWIN_USER_TEMP_DIR"],
            text=True,
            timeout=CONFIG.timeout,
        ).strip()
        return result
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "//tmp"
    except Exception:
        return tempfile.gettempdir()


def check_if_screen_command_available() -> bool:
    try:
        subprocess.run(
            ["which", "screen"],
            capture_output=True,
            check=True,
            timeout=CONFIG.timeout,
        )

        home_dir = os.path.expanduser("~")
        screenrc_path = os.path.join(home_dir, ".screenrc")

        if not os.path.exists(screenrc_path):
            screenrc_content = """defscrollback 10000
termcapinfo xterm* ti@:te@
"""
            with open(screenrc_path, "w") as f:
                f.write(screenrc_content)

        return True
    except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError):
        return False


def get_wcgw_screen_sessions() -> list[str]:
    """Get a list of all WCGW screen session IDs."""
    screen_sessions = []

    try:
        result = subprocess.run(
            ["screen", "-ls"],
            capture_output=True,
            text=True,
            check=False,
            timeout=0.5,
        )
        output = result.stdout or result.stderr or ""

        for line in output.splitlines():
            line = line.strip()
            if not line or not line[0].isdigit():
                continue

            session_parts = line.split()
            if not session_parts:
                continue

            session_id = session_parts[0].strip()

            if ".wcgw." in session_id:
                screen_sessions.append(session_id)
    except Exception:
        pass

    return screen_sessions


def get_orphaned_wcgw_screens() -> list[str]:
    """Identify orphaned WCGW screen sessions where the parent process has PID 1 or doesn't exist."""
    orphaned_screens = []

    try:
        screen_sessions = get_wcgw_screen_sessions()

        for session_id in screen_sessions:
            try:
                pid = int(session_id.split(".")[0])

                try:
                    process = psutil.Process(pid)
                    parent_pid = process.ppid()

                    if parent_pid == 1:
                        orphaned_screens.append(session_id)
                except psutil.NoSuchProcess:
                    orphaned_screens.append(session_id)
            except (ValueError, IndexError):
                continue
    except Exception:
        pass

    return orphaned_screens


def cleanup_orphaned_wcgw_screens(console: Console) -> None:
    """Clean up all orphaned WCGW screen sessions."""
    orphaned_sessions = get_orphaned_wcgw_screens()

    if not orphaned_sessions:
        return

    console.log(
        f"Found {len(orphaned_sessions)} orphaned WCGW screen sessions to clean up"
    )

    for session in orphaned_sessions:
        try:
            subprocess.run(
                ["screen", "-S", session, "-X", "quit"],
                check=False,
                timeout=CONFIG.timeout,
            )
        except Exception as e:
            console.log(f"Failed to kill orphaned screen session: {session}\n{e}")


def cleanup_all_screens_with_name(name: str, console: Console) -> None:
    """Clear all screens with the given name."""
    try:
        result = subprocess.run(
            ["screen", "-ls"],
            capture_output=True,
            text=True,
            check=True,
            timeout=CONFIG.timeout,
        )
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = (e.stdout or "") + (e.stderr or "")
    except FileNotFoundError:
        return
    except Exception as e:
        console.log(f"{e}: exception while clearing running screens.")
        return

    sessions_to_kill = []

    for line in output.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue

        session_info = line.split()[0].strip()
        if session_info.endswith(f".{name}"):
            sessions_to_kill.append(session_info)

    for session in sessions_to_kill:
        try:
            subprocess.run(
                ["screen", "-S", session, "-X", "quit"],
                check=True,
                timeout=CONFIG.timeout,
            )
        except Exception as e:
            console.log(f"Failed to kill screen session: {session}\n{e}")


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


def ensure_wcgw_block_in_rc_file(shell_path: str, console: Console) -> None:
    """Ensure the WCGW environment block exists in the appropriate rc file."""
    rc_file_path = get_rc_file_path(shell_path)
    if not rc_file_path:
        return

    shell_name = os.path.basename(shell_path)

    marker_start = "# --WCGW_ENVIRONMENT_START--"
    marker_end = "# --WCGW_ENVIRONMENT_END--"

    if shell_name == "zsh":
        wcgw_block = f"""{marker_start}
if [ -n "$IN_WCGW_ENVIRONMENT" ]; then
 PROMPT_COMMAND='printf "◉ $(pwd)──➤ \\r\\e[2K"'
 prmptcmdwcgw() {{ eval "$PROMPT_COMMAND" }}
 add-zsh-hook -d precmd prmptcmdwcgw
 precmd_functions+=prmptcmdwcgw
fi
{marker_end}
"""
    elif shell_name == "bash":
        wcgw_block = f"""{marker_start}
if [ -n "$IN_WCGW_ENVIRONMENT" ]; then
 PROMPT_COMMAND='printf "◉ $(pwd)──➤ \\r\\e[2K"'
fi
{marker_end}
"""
    else:
        return

    if not os.path.exists(rc_file_path):
        try:
            with open(rc_file_path, "w") as f:
                f.write(wcgw_block)
            console.log(f"Created {rc_file_path} with WCGW environment block")
        except Exception as e:
            console.log(f"Failed to create {rc_file_path}: {e}")
        return

    try:
        with open(rc_file_path) as f:
            content = f.read()

        if marker_start in content:
            return

        with open(rc_file_path, "a") as f:
            f.write("\n" + wcgw_block)
        console.log(f"Added WCGW environment block to {rc_file_path}")
    except Exception as e:
        console.log(f"Failed to update {rc_file_path}: {e}")


def start_shell(
    is_restricted_mode: bool,
    initial_dir: str,
    console: Console,
    over_screen: bool,
    shell_path: str,
) -> tuple["pexpect.spawn[str]", str]:
    cmd = shell_path
    if is_restricted_mode and cmd.split("/")[-1] == "bash":
        cmd += " -r"

    overrideenv = {
        **os.environ,
        "PROMPT_COMMAND": PROMPT_COMMAND,
        "TMPDIR": get_tmpdir(),
        "TERM": "xterm-256color",
        "IN_WCGW_ENVIRONMENT": "1",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
    }
    try:
        shell = pexpect.spawn(
            cmd,
            env=overrideenv,
            echo=True,
            encoding="utf-8",
            timeout=CONFIG.timeout,
            cwd=initial_dir,
            codec_errors="backslashreplace",
            dimensions=(500, 160),
        )
        shell.sendline(PROMPT_STATEMENT)
        shell.expect(PROMPT_CONST, timeout=CONFIG.timeout)
    except Exception as e:
        console.print(traceback.format_exc())
        console.log(f"Error starting shell: {e}. Retrying without rc ...")

        shell = pexpect.spawn(
            "/bin/bash --noprofile --norc",
            env=overrideenv,
            echo=True,
            encoding="utf-8",
            timeout=CONFIG.timeout,
            codec_errors="backslashreplace",
        )
        shell.sendline(PROMPT_STATEMENT)
        shell.expect(PROMPT_CONST, timeout=CONFIG.timeout)

    initialdir_hash = md5(
        os.path.normpath(os.path.abspath(initial_dir)).encode()
    ).hexdigest()[:5]
    shellid = shlex.quote(
        "wcgw."
        + time.strftime("%d-%Hh%Mm%Ss")
        + f".{initialdir_hash[:3]}."
        + os.path.basename(initial_dir)
    )
    if over_screen:
        if not check_if_screen_command_available():
            raise ValueError("Screen command not available")
        while True:
            output = shell.expect([PROMPT_CONST, pexpect.TIMEOUT], timeout=0.1)
            if output == 1:
                break
        shell.sendline(f"screen -q -S {shellid} {shell_path}")
        shell.expect(PROMPT_CONST, timeout=CONFIG.timeout)

    return shell, shellid


def render_terminal_output(text: str) -> list[str]:
    screen = pyte.Screen(160, 500)
    screen.set_mode(pyte_modes.LNM)
    stream = pyte.Stream(screen)
    stream.feed(text)
    # Filter out empty lines
    dsp = screen.display[::-1]
    for i, line in enumerate(dsp):
        if line.strip():
            break
    else:
        i = len(dsp)
    lines = screen.display[: len(dsp) - i]
    return lines
