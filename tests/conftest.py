"""Shared test configuration.

The shell's command wait budget (CONFIG.timeout) defaults to 5s; the MCP server
runs with 2s (BABASH_TIMEOUT). Pin the tests to the same short budget so
"still running" is deterministic for commands longer than it, and the suite
doesn't idle 5s per long command.

Timing contract for tests that exercise running/exited transitions:
- "still running": use a command far longer than CONFIG.timeout (e.g. `sleep 30`)
  and let the fixture cleanup interrupt it.
- "running then exited": use a moderate command (e.g. `sleep 4`) and pass an
  explicit wait_for_seconds on the follow-up status check that comfortably
  exceeds the command duration.
"""

from babash.client.bash_state.shell_process import CONFIG

CONFIG.timeout = 2.0
CONFIG.timeout_while_output = 20.0
CONFIG.output_wait_patience = 3.0
