"""Sending one action to a shell and reading back what it says."""

from __future__ import annotations

import tempfile
import threading
from typing import TYPE_CHECKING, Optional

import pexpect

from ...types_ import (
    BashAction,
    Command,
    SendSpecials,
    SendText,
    Specials,
    StatusCheck,
)
from .shell_process import cleanup_orphaned_babash_screens

if TYPE_CHECKING:
    from .bash_state import BashState


BUSY_MESSAGE = """A command is already running in this session. One session runs one command at a time.
1. Use `check_status` to get its output.
2. Use `send_input` to give text input to the running program.
3. Use `send_keys` with Ctrl-c to kill it.
4. Or run the new command elsewhere: is_background=true, or create_session.
"""

# Every special key babash accepts, and the bytes it sends.
#
# Ctrl-c is absent because it is not bytes: it goes through pexpect's
# `sendintr()`, which looks up the terminal's own interrupt character and
# delivers a signal to the foreground process group.
#
# Ctrl-d is NOT an interrupt, whatever its neighbour in this list suggests. It
# is end-of-input: the byte that closes a program's stdin, exits a REPL, or
# finishes a `cat > file`. Upstream lumped it in with Ctrl-c and sent SIGINT for
# it, so asking a Python REPL to quit merely gave it a KeyboardInterrupt and
# left the agent stuck inside.
#
# Keying this on `Specials` means the type checker — not a runtime raise in an
# else-branch — is what guarantees every key has a mapping.
_SPECIAL_KEYS: dict[Specials, str] = {
    "Enter": "\x0d",
    "Tab": "\t",
    "Backspace": "\x7f",
    "Escape": "\x1b",
    "Key-up": "\033[A",
    "Key-down": "\033[B",
    "Key-left": "\033[D",
    "Key-right": "\033[C",
    "Home": "\033[H",
    "End": "\033[F",
    "PageUp": "\033[5~",
    "PageDown": "\033[6~",
    "Ctrl-d": "\x04",
    "Ctrl-z": "\x1a",
    "Ctrl-l": "\x0c",
}
_INTERRUPT_KEYS: frozenset[Specials] = frozenset({"Ctrl-c"})

# execute_bash's reply is "<output>" + this + "<status>".
STATUS_SEPARATOR = "\n\n---\n\n"


def get_status(shell: "BashState") -> str:
    status = STATUS_SEPARATOR
    if shell.state == "pending":
        prompt = shell.pending_prompt()
        if prompt:
            # Say this plainly, or the agent reads "still running" and settles in
            # to poll a command that will never move until it is answered.
            status += "status = waiting for input\n"
            status += f"prompt = {prompt!r}\n"
            status += "Answer it with send_input(text=...), or send_keys for control keys.\n"
        else:
            status += "status = still running\n"
        status += "running for = " + shell.get_pending_for() + "\n"
    else:
        status += "status = process exited\n"
        if shell.last_exit_code is not None:
            status += f"exit code = {shell.last_exit_code}\n"
    status += "cwd = " + shell.cwd + "\n"
    return status.rstrip()


def is_status_check(action: BashAction) -> bool:
    """Whether this call is only asking "what's happened since I last looked?"

    A bare Enter counts: it advances nothing, so the caller is really just
    waiting on output, and gets the longer budget.
    """
    return isinstance(action, StatusCheck) or (
        isinstance(action, SendSpecials) and action.send_specials == ("Enter",)
    )


def _send_action(shell: "BashState", action: BashAction) -> str | None:
    """Write the action to the shell.

    Returns a message to hand straight back to the caller if there was nothing
    to send, or None once the input is on its way.
    """
    if isinstance(action, Command):
        if shell.state == "pending":
            return BUSY_MESSAGE

        shell.console.print(f"$ {action.command}")
        command = action.command.strip()
        shell.clear_to_run()
        # Chunked: a pty's input buffer is finite, and a long line written in
        # one go can be silently truncated.
        for i in range(0, len(command), 64):
            shell.send(command[i : i + 64], set_as_command=None)
        shell.send(shell.linesep, set_as_command=command)
        return None

    if isinstance(action, StatusCheck):
        shell.console.print("Checking status")
        if shell.state != "pending":
            return "No running command to check status of."
        return None

    if isinstance(action, SendText):
        if not action.send_text:
            return "Failure: send_text cannot be empty. Use send_keys('Enter') instead."
        shell.console.print(f"Interact text: {action.send_text!r}")
        for i in range(0, len(action.send_text), 128):
            shell.send(action.send_text[i : i + 128], set_as_command=None)
        shell.send(shell.linesep, set_as_command=None)
        return None

    if not action.send_specials:
        return "Failure: send_keys cannot be empty"
    shell.console.print(f"Sending special sequence: {action.send_specials}")
    for key in action.send_specials:
        if key in _INTERRUPT_KEYS:
            shell.sendintr()
        else:
            shell.send(_SPECIAL_KEYS[key], set_as_command=None)
    return None


def _is_interrupt(action: BashAction) -> bool:
    """Whether this was an attempt to kill what's running — which is the only
    thing the "couldn't interrupt it" advice makes sense for."""
    return isinstance(action, SendSpecials) and any(
        key in _INTERRUPT_KEYS for key in action.send_specials
    )


def _wait_for_output(
    shell: "BashState", is_check: bool, timeout_s: Optional[float]
) -> tuple[str, bool]:
    """Read until the shell prompts again or the budget runs out.

    Returns the new output and whether the command finished.

    `timeout_s` is the caller's budget; without one, the shell's default applies.
    A plain run_command takes that default: quick commands (most of them) come
    back immediately, and anything slower returns "pending" so the agent can
    decide what to do rather than sit blocked. A caller that already knows it
    isn't going to wait — a background command — passes a small budget instead,
    just big enough to catch a command that dies on the spot.

    Only a check is worth spending the budget in slices: it is the call that is
    explicitly waiting, so it keeps going until several slices in a row have
    produced nothing, on the grounds that a command which has gone quiet is
    unlikely to speak up again.
    """
    timings = shell.timings
    total_budget = float(timeout_s) if timeout_s else timings.command_budget
    slice_wait = min(total_budget, timings.output_slice)

    if shell.expect([shell.prompt, pexpect.TIMEOUT], timeout=slice_wait) == 0:
        return shell.incremental_output(), True

    collected = [shell.incremental_output()]
    remaining = total_budget - slice_wait
    if not is_check:
        return _settle(shell, collected)

    patience = timings.quiet_slices_before_giving_up
    if not collected[0]:
        patience -= 1

    while remaining > 0 and patience > 0:
        this_wait = min(remaining, timings.output_slice)
        finished = shell.expect([shell.prompt, pexpect.TIMEOUT], timeout=this_wait) == 0
        collected.append(shell.incremental_output())
        if finished:
            shell.set_awaiting_input(None)
            return _join(collected), True
        patience = timings.quiet_slices_before_giving_up if collected[-1] else patience - 1
        remaining -= this_wait

    return _settle(shell, collected)


def _settle(shell: "BashState", collected: list[str]) -> tuple[str, bool]:
    """Hand back a still-running command, having first checked whether it is
    running at all or is standing there waiting to be answered."""
    extra, finished = _detect_pending_prompt(shell)
    collected.append(extra)
    return _join(collected), finished


def _join(chunks: list[str]) -> str:
    return "\n".join(chunk for chunk in chunks if chunk)


# How long to watch a parked cursor before calling it a prompt. Long enough that
# anything still working — a progress bar, a spinner — gives itself away by
# writing again; short enough not to be felt.
PROMPT_SETTLE_SECONDS = 0.5


def _detect_pending_prompt(shell: "BashState") -> tuple[str, bool]:
    """Work out whether the command has stopped to ask something.

    A cursor parked mid-line is the signature of a program blocked in read() —
    it printed `Select [1/2/3]: ` and is waiting. But it is also the signature of
    a progress bar mid-redraw, so the cursor alone proves nothing. The tell is
    what happens next: a working program writes again, a blocked one is silent.
    So we wait a beat and look.

    Returns any output that arrived during that beat (which must not be dropped)
    and whether the command finished while we watched. The verdict itself is
    recorded on the shell, so the session roster can report it too.
    """
    if shell.cursor_prompt() is None:
        shell.set_awaiting_input(None)
        return "", False

    finished = shell.expect([shell.prompt, pexpect.TIMEOUT], timeout=PROMPT_SETTLE_SECONDS) == 0
    extra = shell.incremental_output()
    if finished:
        shell.set_awaiting_input(None)
        return extra, True

    # It spoke while we watched, so it isn't waiting on us.
    shell.set_awaiting_input(None if extra else shell.cursor_prompt())
    return extra, False


def truncate(output: str, max_chars: Optional[int]) -> str:
    """Keep the tail of an over-long output, spilling the whole thing to a file.

    The tail, not the head: the end of a build log is where the error is.

    This limit used to be measured in LLM tokens, which meant shipping a 9MB
    tokenizer and fetching a vocabulary from huggingface.co on first use — a
    network round-trip, at startup, in a shell server, to decide where to cut a
    string. Characters are a fine proxy for the only thing the limit is for:
    not flooding the agent's context.
    """
    if not max_chars or len(output) <= max_chars:
        return output

    saved = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    saved.write(output.encode())
    saved.close()
    return (
        f"(...OUTPUT TRUNCATED — {len(output)} chars, showing the last {max_chars}. "
        f"Full output saved to {saved.name}. TIP: use more precise commands "
        f"(grep, head, tail, awk) instead of dumping everything.)\n" + output[-max_chars:]
    )


def _reply(shell: "BashState", action: BashAction, output: str, finished: bool) -> str:
    """Assemble what the agent sees: the output, then the shell's standing."""
    if _is_interrupt(action) and not finished:
        output += (
            "\n---\n----\nFailure interrupting.\n"
            "You may want to try Ctrl-c again or program specific exit interactive commands.\n"
        )

    if isinstance(action, Command):
        # The pty echoes what we typed back at us; the agent sent it, so it
        # knows. Drop it rather than pay context to repeat it.
        command = action.command.strip()
        if output.startswith(command):
            output = output[len(command) :]

    return output + get_status(shell)


def execute_bash(
    shell: "BashState",
    action: BashAction,
    max_chars: Optional[int],
    timeout_s: Optional[float],
) -> str:
    """Send one action to a shell and read back whatever it has to say."""
    try:
        try:
            early_reply = _send_action(shell, action)
        except KeyboardInterrupt:
            shell.sendintr()
            shell.expect(shell.prompt)
            return "---\n\nFailure: user interrupted the execution"
        if early_reply is not None:
            return early_reply

        output, finished = _wait_for_output(shell, is_status_check(action), timeout_s)
        # Order matters: _wait_for_output reads the renderer's "what's new"
        # state, and set_repl clears it.
        if finished:
            shell.set_repl()
        else:
            shell.set_pending()

        return _reply(shell, action, truncate(output, max_chars), finished)
    finally:
        shell.start_idle_reader()
        if shell.over_screen:
            threading.Thread(
                target=cleanup_orphaned_babash_screens,
                args=(shell.console,),
                daemon=True,
            ).start()
