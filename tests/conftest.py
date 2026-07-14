"""Shared test fixtures.

Timings are passed to the shell, not patched into a global. They used to be the
latter, and the fixture had to explain that the module default (5s) and the one
the server actually ran with (2s) disagreed — which is precisely the confusion
`ShellTimings` exists to remove. The tests now say what they want and get it.

Timing contract for tests that exercise running/exited transitions:
- "still running": use a command far longer than `command_budget` (e.g. `sleep 30`)
  and let the fixture's cleanup interrupt it.
- "running then exited": use a moderate command (e.g. `sleep 4`) and pass an
  explicit wait that comfortably exceeds the command's duration.
"""

import tempfile
from typing import Any, Iterator

import pytest

from babash.client.bash_state import BashState
from babash.settings import ShellTimings

TEST_TIMINGS = ShellTimings(
    command_budget=2.0,
    output_slice=20.0,
    quiet_slices_before_giving_up=3.0,
)


class QuietConsole:
    def print(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log(self, *args: Any, **kwargs: Any) -> None:
        pass


@pytest.fixture
def temp_dir() -> Iterator[str]:
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
def shell(temp_dir: str) -> Iterator[BashState]:
    """A real shell in a throwaway directory.

    use_screen=False deliberately: under `screen` the pty attaches to the
    terminal running pytest, which takes over the developer's session.
    """
    state = BashState(
        console=QuietConsole(),
        working_dir=temp_dir,
        use_screen=False,
        shell_id=None,
        shell_path=None,
        timings=TEST_TIMINGS,
    )
    try:
        yield state
    finally:
        try:
            state.sendintr()
        except Exception:
            pass
        state.cleanup()
