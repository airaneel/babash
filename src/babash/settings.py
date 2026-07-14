"""Everything babash reads from its environment, read in one place.

This used to be a mutable `CONFIG` singleton that the server's lifespan reached
out and rewrote at startup, that the tests reached out and rewrote too, and that
twenty call sites deep in the shell layer read back. Three consequences, all of
which this exists to end:

- The same knob had two different defaults depending on who was asking — the
  dataclass said the command budget was 5 seconds, the lifespan overrode it to 2
  — so what a shell actually did depended on whether a server had started yet.
- One value, `timeout`, was doing three unrelated jobs: how long to wait for a
  command before calling it pending, how long to give a `screen -ls` subprocess,
  and how long to allow a shell to boot and source its rc file. Turning the first
  one down for responsiveness quietly made the third one fragile.
- Settings arrived by mutation rather than by argument, so nothing in a
  signature said a shell's behaviour depended on them.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ShellTimings:
    """How long to wait on a shell before concluding something about it."""

    command_budget: float
    """How long a command gets before we hand back "still running" and let the
    agent decide what to do. Short on purpose: most commands beat it, and the
    ones that don't are better polled than blocked on."""

    output_slice: float
    """The size of each slice a check_status spends its budget in."""

    quiet_slices_before_giving_up: float
    """How many slices in a row may produce nothing before a check_status stops
    waiting. A command that has gone silent is unlikely to speak up again."""

    @staticmethod
    def from_env() -> "ShellTimings":
        return ShellTimings(
            command_budget=float(os.getenv("BABASH_TIMEOUT", "2")),
            output_slice=float(os.getenv("BABASH_TIMEOUT_WHILE_OUTPUT", "15")),
            quiet_slices_before_giving_up=float(os.getenv("BABASH_OUTPUT_PATIENCE", "3")),
        )


@dataclass(frozen=True)
class Settings:
    shell_path: Optional[str]
    """The shell to spawn. None means $SHELL, or /bin/bash."""

    workspace: str
    """Where a chat's shell starts out, absent anything more specific."""

    max_output_chars: int
    """How much of a command's output the agent gets back. Measured in
    characters: the point of the limit is not to flood the agent's context, and
    characters track that closely enough without dragging in a tokenizer."""

    timings: ShellTimings
    host: str
    port: int
    debug: bool

    @staticmethod
    def from_env(tmp_dir: str) -> "Settings":
        """Read once, at import. There is no setter and nothing overrides this
        later: the environment is the single way to configure babash, so there
        is exactly one place a given knob can have come from."""
        return Settings(
            shell_path=os.getenv("BABASH_SHELL") or None,
            workspace=os.path.join(tmp_dir, "claude_playground"),
            max_output_chars=int(os.getenv("BABASH_MAX_OUTPUT_CHARS", "60000")),
            timings=ShellTimings.from_env(),
            host=os.getenv("BABASH_HOST", "127.0.0.1"),
            port=int(os.getenv("BABASH_PORT", "8000")),
            debug=bool(os.getenv("BABASH_DEBUG")),
        )
