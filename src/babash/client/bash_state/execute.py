from __future__ import annotations

import tempfile
import threading
import traceback
from typing import TYPE_CHECKING, Optional

import pexpect  # type: ignore[import-untyped]

from ...types_ import (
    BashCommand,
    Command,
    SendAscii,
    SendSpecials,
    SendText,
    StatusCheck,
)
from ..encoder import EncoderDecoder
from .parser.bash_statement_parser import BashStatementParser
from .shell_process import (
    CONFIG,
    cleanup_orphaned_babash_screens,
    render_terminal_output,
)

if TYPE_CHECKING:
    from .bash_state import BashState


WAITING_INPUT_MESSAGE = """A command is already running. NOTE: You can't run multiple shell commands in main shell, likely a previous program hasn't exited.
1. Get its output using status check.
2. Use `send_ascii` or `send_specials` to give inputs to the running program OR
3. kill the previous program by sending ctrl+c first using `send_ascii` or `send_specials`
4. Interrupt and run the process in background
"""


def get_incremental_output(old_output: list[str], new_output: list[str]) -> list[str]:
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


def rstrip(lines: list[str]) -> str:
    return "\n".join([line.rstrip() for line in lines])


def _incremental_text(text: str, last_pending_output: str) -> str:
    text = text[-100_000:]

    if not last_pending_output:
        return rstrip(render_terminal_output(text)).lstrip()
    last_rendered_lines = render_terminal_output(last_pending_output)
    last_pending_output_rendered = "\n".join(last_rendered_lines)
    if not last_rendered_lines:
        return rstrip(render_terminal_output(text))

    text = text[len(last_pending_output) :]
    old_rendered_applied = render_terminal_output(last_pending_output_rendered + text)
    rendered = get_incremental_output(last_rendered_lines[:-1], old_rendered_applied)

    if not rendered:
        return ""

    if rendered[0] == last_rendered_lines[-1]:
        rendered = rendered[1:]
    return rstrip(rendered)


def get_status(bash_state: "BashState", is_bg: bool) -> str:
    status = "\n\n---\n\n"
    if is_bg:
        status += f"bg_command_id = {bash_state.current_thread_id}\n"
    if bash_state.state == "pending":
        status += "status = still running\n"
        status += "running for = " + bash_state.get_pending_for() + "\n"
        status += "cwd = " + bash_state.cwd + "\n"
    else:
        bg_desc = ""
        status += "status = process exited" + bg_desc + "\n"
        status += "cwd = " + bash_state.cwd + "\n"

    if not is_bg:
        status += "This is the main shell. " + get_bg_running_commandsinfo(bash_state)

    return status.rstrip()


def is_status_check(arg: BashCommand) -> bool:
    return (
        isinstance(arg.action_json, StatusCheck)
        or (
            isinstance(arg.action_json, SendSpecials)
            and arg.action_json.send_specials == ["Enter"]
        )
        or (
            isinstance(arg.action_json, SendAscii)
            and arg.action_json.send_ascii == [10]
        )
    )


def execute_bash(
    bash_state: "BashState",
    enc: EncoderDecoder[int],
    bash_arg: BashCommand,
    max_tokens: Optional[int],
    timeout_s: Optional[float],
) -> tuple[str, float]:
    try:
        # Check if the thread_id matches current
        if bash_arg.action_json.thread_id != bash_state.current_thread_id:
            if not bash_state.load_state_from_thread_id(bash_arg.action_json.thread_id):
                return (
                    f"Error: No saved bash state found for thread_id `{bash_arg.action_json.thread_id}`. Please initialize first with this ID.",
                    0.0,
                )

        output, cost = _execute_bash(bash_state, enc, bash_arg, max_tokens, timeout_s)

        # Remove echo if it's a command
        if isinstance(bash_arg.action_json, Command):
            command = bash_arg.action_json.command.strip()
            if output.startswith(command):
                output = output[len(command) :]

    finally:
        bash_state.run_bg_expect_thread()
        if bash_state.over_screen:
            thread = threading.Thread(
                target=cleanup_orphaned_babash_screens,
                args=(bash_state.console,),
                daemon=True,
            )
            thread.start()
    return output, cost


def assert_single_statement(command: str) -> None:
    if "\n" in command:
        try:
            parser = BashStatementParser()
            statements = parser.parse_string(command)
        except Exception:
            raise ValueError(
                "Command should not contain newline character in middle. Run only one command at a time."
            )
        if len(statements) > 1:
            raise ValueError(
                "Error: Command contains multiple statements. Please run only one bash statement at a time."
            )


def get_bg_running_commandsinfo(bash_state: "BashState") -> str:
    msg = ""
    running = []
    for id_, state in bash_state.background_shells.items():
        running.append(f"Command: {state.last_command}, bg_command_id: {id_}")
    if running:
        msg = (
            "Following background commands are attached:\n" + "\n".join(running) + "\n"
        )
    else:
        msg = "No command running in background.\n"
    return msg


def _execute_bash(
    bash_state: "BashState",
    enc: EncoderDecoder[int],
    bash_arg: BashCommand,
    max_tokens: Optional[int],
    timeout_s: Optional[float],
) -> tuple[str, float]:
    try:
        is_interrupt = False
        command_data = bash_arg.action_json
        is_bg = False
        og_bash_state = bash_state

        if not isinstance(command_data, Command) and command_data.bg_command_id:
            if command_data.bg_command_id not in bash_state.background_shells:
                error = f"No shell found running with command id {command_data.bg_command_id}.\n"
                if bash_state.background_shells:
                    error += get_bg_running_commandsinfo(bash_state)
                if bash_state.state == "pending":
                    error += f"On the main thread a command is already running ({bash_state.last_command})"
                else:
                    error += "On the main thread no command is running."
                raise Exception(error)
            bash_state = bash_state.background_shells[command_data.bg_command_id]
            is_bg = True

        if isinstance(command_data, Command):
            if bash_state.bash_command_mode.allowed_commands == "none":
                return "Error: BashCommand not allowed in current mode", 0.0

            bash_state.console.print(f"$ {command_data.command}")

            command = command_data.command.strip()

            assert_single_statement(command)

            if command_data.is_background:
                bash_state = bash_state.start_new_bg_shell(bash_state.cwd)
                is_bg = True

            if bash_state.state == "pending":
                raise ValueError(WAITING_INPUT_MESSAGE)

            bash_state.clear_to_run()
            for i in range(0, len(command), 64):
                bash_state.send(command[i : i + 64], set_as_command=None)
            bash_state.send(bash_state.linesep, set_as_command=command)
        elif isinstance(command_data, StatusCheck):
            bash_state.console.print("Checking status")
            if bash_state.state != "pending":
                error = "No running command to check status of.\n"
                error += get_bg_running_commandsinfo(bash_state)
                return error, 0.0

        elif isinstance(command_data, SendText):
            if not command_data.send_text:
                return "Failure: send_text cannot be empty", 0.0

            bash_state.console.print(f"Interact text: {command_data.send_text}")
            for i in range(0, len(command_data.send_text), 128):
                bash_state.send(
                    command_data.send_text[i : i + 128], set_as_command=None
                )
            bash_state.send(bash_state.linesep, set_as_command=None)

        elif isinstance(command_data, SendSpecials):
            if not command_data.send_specials:
                return "Failure: send_specials cannot be empty", 0.0

            bash_state.console.print(
                f"Sending special sequence: {command_data.send_specials}"
            )
            for char in command_data.send_specials:
                if char == "Key-up":
                    bash_state.send("\033[A", set_as_command=None)
                elif char == "Key-down":
                    bash_state.send("\033[B", set_as_command=None)
                elif char == "Key-left":
                    bash_state.send("\033[D", set_as_command=None)
                elif char == "Key-right":
                    bash_state.send("\033[C", set_as_command=None)
                elif char == "Enter":
                    bash_state.send("\x0d", set_as_command=None)
                elif char == "Ctrl-c":
                    bash_state.sendintr()
                    is_interrupt = True
                elif char == "Ctrl-d":
                    bash_state.sendintr()
                    is_interrupt = True
                elif char == "Ctrl-z":
                    bash_state.send("\x1a", set_as_command=None)
                else:
                    raise Exception(f"Unknown special character: {char}")

        elif isinstance(command_data, SendAscii):
            if not command_data.send_ascii:
                return "Failure: send_ascii cannot be empty", 0.0

            bash_state.console.print(
                f"Sending ASCII sequence: {command_data.send_ascii}"
            )
            for ascii_char in command_data.send_ascii:
                bash_state.send(chr(ascii_char), set_as_command=None)
                if ascii_char == 3:
                    is_interrupt = True
        else:
            raise ValueError(f"Unknown command type: {type(command_data)}")

    except KeyboardInterrupt:
        bash_state.sendintr()
        bash_state.expect(bash_state.prompt)
        return "---\n\nFailure: user interrupted the execution", 0.0

    wait = min(timeout_s or CONFIG.timeout, CONFIG.timeout_while_output)
    index = bash_state.expect([bash_state.prompt, pexpect.TIMEOUT], timeout=wait)
    if index == 1:
        text = bash_state.before or ""
        incremental_text = _incremental_text(text, bash_state.pending_output)

        second_wait_success = False
        if is_status_check(bash_arg):
            remaining = CONFIG.timeout_while_output - wait
            patience = CONFIG.output_wait_patience
            if not incremental_text:
                patience -= 1
            itext = incremental_text
            while remaining > 0 and patience > 0:
                index = bash_state.expect(
                    [bash_state.prompt, pexpect.TIMEOUT], timeout=wait
                )
                if index == 0:
                    second_wait_success = True
                    break
                else:
                    _itext = bash_state.before or ""
                    _itext = _incremental_text(_itext, bash_state.pending_output)
                    if _itext != itext:
                        patience = 3
                    else:
                        patience -= 1
                    itext = _itext

                remaining = remaining - wait

            if not second_wait_success:
                text = bash_state.before or ""
                incremental_text = _incremental_text(text, bash_state.pending_output)

        if not second_wait_success:
            bash_state.set_pending(text)

            tokens = enc.encoder(incremental_text)

            if max_tokens and len(tokens) >= max_tokens:
                saved = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
                saved.write(incremental_text.encode())
                saved.close()
                incremental_text = (
                    f"(...truncated, full output saved to {saved.name})\n"
                    + enc.decoder(tokens[-(max_tokens - 1) :])
                )

            if is_interrupt:
                incremental_text = (
                    incremental_text
                    + """---
----
Failure interrupting.
You may want to try Ctrl-c again or program specific exit interactive commands.
    """
                )

            exit_status = get_status(bash_state, is_bg)
            incremental_text += exit_status
            if is_bg and bash_state.state == "repl":
                try:
                    bash_state.cleanup()
                    og_bash_state.background_shells.pop(bash_state.current_thread_id)
                except Exception as e:
                    bash_state.console.log(f"error while cleaning up {e}")

            return incremental_text, 0

    before = str(bash_state.before)

    output = _incremental_text(before, bash_state.pending_output)
    bash_state.set_repl()

    tokens = enc.encoder(output)
    if max_tokens and len(tokens) >= max_tokens:
        saved = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        saved.write(output.encode())
        saved.close()
        output = (
            f"(...truncated, full output saved to {saved.name})\n"
            + enc.decoder(tokens[-(max_tokens - 1) :])
        )

    try:
        exit_status = get_status(bash_state, is_bg)
        output += exit_status
        if is_bg and bash_state.state == "repl":
            try:
                bash_state.cleanup()
                og_bash_state.background_shells.pop(bash_state.current_thread_id)
            except Exception as e:
                bash_state.console.log(f"error while cleaning up {e}")
    except ValueError:
        bash_state.console.print(output)
        bash_state.console.print(traceback.format_exc())
        bash_state.console.print("Malformed output, restarting shell", style="red")
        bash_state.reset_shell()
        output = "(exit shell has restarted)"
    return output, 0
