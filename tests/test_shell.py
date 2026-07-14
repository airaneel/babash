"""Running things in a shell and finding out what happened."""

import time

from babash.client.bash_state import BashState, execute_bash
from babash.types_ import Command, SendSpecials, SendText, StatusCheck


def run(shell: BashState, command: str) -> str:
    return execute_bash(shell, Command(command=command), None, None)


def test_command_output(shell: BashState) -> None:
    assert "hello-from-babash" in run(shell, "echo hello-from-babash")


def test_exit_code_is_reported(shell: BashState) -> None:
    assert "exit code = 0" in run(shell, "true")
    assert "exit code = 42" in run(shell, "(exit 42)")
    assert "exit code = 1" in run(shell, "ls /definitely/not/here")


def test_cwd_follows_the_shell(shell: BashState) -> None:
    run(shell, "mkdir -p subdir && cd subdir")
    assert shell.cwd.endswith("subdir")
    assert "cwd = " in run(shell, "pwd")


def test_env_persists_across_commands(shell: BashState) -> None:
    run(shell, "export MARKER=persisted")
    assert "persisted" in run(shell, "echo $MARKER")


def test_long_command_returns_pending_not_blocked(shell: BashState) -> None:
    started = time.monotonic()
    out = run(shell, "sleep 30")
    elapsed = time.monotonic() - started

    assert "still running" in out
    assert shell.state == "pending"
    # The point of "pending": the agent gets control back rather than blocking
    # for the command's full duration.
    assert elapsed < 10, f"run_command blocked for {elapsed:.1f}s instead of returning pending"


def test_busy_shell_refuses_a_second_command(shell: BashState) -> None:
    run(shell, "sleep 30")
    out = run(shell, "echo should-not-run")
    assert "already running" in out
    assert "should-not-run" not in out


def test_status_check_collects_output_as_it_arrives(shell: BashState) -> None:
    # run_command hands back whatever the command has printed by the time its
    # budget runs out, together with state=pending — so the first chunk arrives
    # from the run itself, and check_status continues from there.
    seen = run(shell, "for i in 1 2 3; do echo tick-$i; sleep 1; done")
    for _ in range(8):
        seen += execute_bash(shell, StatusCheck(), None, 3.0)
        if "process exited" in seen:
            break
    assert "tick-1" in seen
    assert "tick-3" in seen, "later output must keep arriving, not just the first chunk"


def test_status_check_with_nothing_running(shell: BashState) -> None:
    run(shell, "true")
    assert "No running command" in execute_bash(shell, StatusCheck(), None, None)


def test_ctrl_c_interrupts(shell: BashState) -> None:
    run(shell, "sleep 30")
    assert shell.state == "pending"
    execute_bash(shell, SendSpecials(send_specials=("Ctrl-c",)), None, None)
    # The shell is usable again straight away.
    assert "after-interrupt" in run(shell, "echo after-interrupt")


def test_send_input_answers_a_prompt(shell: BashState) -> None:
    run(shell, "read -p 'name: ' answer; echo got-$answer")
    out = execute_bash(shell, SendText(send_text="babash"), None, None)
    assert "got-babash" in out


def test_ctrl_d_sends_eof_not_an_interrupt(shell: BashState) -> None:
    """Ctrl-d closes stdin. Upstream routed it through sendintr(), so asking a
    REPL to quit only handed it a KeyboardInterrupt and left the agent inside."""
    run(shell, "python3 -q")
    assert shell.state == "pending", "we should be inside the REPL"

    execute_bash(shell, SendSpecials(send_specials=("Ctrl-d",)), None, None)
    assert shell.state == "repl", "Ctrl-d must exit the REPL, not interrupt it"
    assert "back-in-bash" in run(shell, "echo back-in-bash")


def test_ctrl_d_ends_a_heredoc_style_read(shell: BashState) -> None:
    run(shell, "cat > eof-test.txt")
    execute_bash(shell, SendText(send_text="written via stdin"), None, None)
    execute_bash(shell, SendSpecials(send_specials=("Ctrl-d",)), None, None)
    assert "written via stdin" in run(shell, "cat eof-test.txt")


def test_every_special_key_has_a_mapping() -> None:
    """The Literal and the keymap must not drift apart."""
    from typing import get_args

    from babash.client.bash_state.execute import _INTERRUPT_KEYS, _SPECIAL_KEYS
    from babash.types_ import Specials

    assert set(get_args(Specials)) == set(_SPECIAL_KEYS) | set(_INTERRUPT_KEYS)


def test_empty_input_is_rejected_with_advice(shell: BashState) -> None:
    assert "cannot be empty" in execute_bash(shell, SendText(send_text=""), None, None)
    assert "cannot be empty" in execute_bash(
        shell, SendSpecials(send_specials=()), None, None
    )


def test_output_is_truncated_from_the_tail(shell: BashState) -> None:
    """The end of a build log is where the error is, so keep the end."""
    out = execute_bash(shell, Command(command="seq 1 5000"), 200, None)
    assert "OUTPUT TRUNCATED" in out
    assert "5000" in out, "the tail must be kept"
    assert "\n1\n" not in out, "the head must be dropped"


def test_large_output_keeps_flowing_past_100kb(shell: BashState) -> None:
    """A command that prints far more than the old 100KB re-render cap.

    The previous renderer went silent past that point (see TerminalRenderer);
    this pins the end-to-end behaviour, not just the unit.
    """
    seen = run(shell, "seq 1 3000 | awk '{printf \"%s-PADDING-PADDING-PADDING\\n\", $0}'")
    for _ in range(8):
        seen += execute_bash(shell, StatusCheck(), None, 3.0)
        if "process exited" in seen:
            break
    assert "3000-PADDING" in seen, "output must still arrive after the buffer gets big"


def test_a_blocked_prompt_is_reported_as_waiting_for_input(shell: BashState) -> None:
    """The distinction an agent cannot make on its own.

    "still running" and "waiting for you" both look like a shell that has gone
    quiet — so an agent told the former polls a command that will never move.
    """
    out = run(shell, "read -p 'Select [1/2/3]: ' choice; echo picked $choice")
    assert "waiting for input" in out
    assert "Select [1/2/3]:" in out
    assert shell.pending_prompt() == "Select [1/2/3]:"

    out = execute_bash(shell, SendText(send_text="2"), None, None)
    assert "picked 2" in out
    assert shell.pending_prompt() is None


def test_a_progress_bar_is_not_mistaken_for_a_prompt(shell: BashState) -> None:
    """A redrawing progress bar parks the cursor mid-line exactly like a prompt
    does. What separates them is that it keeps writing — so we watch for a beat."""
    out = run(shell, r'for i in $(seq 1 40); do printf "\rWorking: %s%%" $i; sleep 0.3; done')
    assert "waiting for input" not in out
    assert "still running" in out
    assert shell.pending_prompt() is None


def test_a_merely_slow_command_is_not_a_prompt(shell: BashState) -> None:
    out = run(shell, "sleep 30")
    assert "still running" in out
    assert "waiting for input" not in out
    assert shell.pending_prompt() is None


def test_password_prompt_with_echo_off(shell: BashState) -> None:
    """`read -s` prints the prompt but echoes nothing typed — the cursor still
    parks after it, so this must be caught too."""
    out = run(shell, "read -sp 'Password: ' pw; echo; echo len=${#pw}")
    assert "waiting for input" in out

    out = execute_bash(shell, SendText(send_text="hunter2"), None, None)
    assert "len=7" in out
    assert "hunter2" not in out, "the password must not be echoed back"
