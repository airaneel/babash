"""Output processing and shared utilities."""

import os

from mcp.server.fastmcp import Context as McpContext

from ..tools import Context, initialize
from .state import AppState, CommandRecord

CODING_MAX_TOKENS = int(os.getenv("BABASH_CODING_MAX_TOKENS", "32000"))
NONCODING_MAX_TOKENS = int(os.getenv("BABASH_NONCODING_MAX_TOKENS", "16000"))

_ERROR_PATTERNS: list[tuple[str, str]] = [
    ("command not found", "Command not installed. Try: apt install <package> or brew install <package>"),
    ("No such file or directory", "Path doesn't exist. Check with: ls <parent_dir>"),
    ("Permission denied", "No permission. Try: sudo <command> or check file permissions with ls -la"),
    ("Connection refused", "Service not running or wrong port. Check with: ss -tlnp or systemctl status"),
    ("Connection timed out", "Host unreachable. Check network with: ping <host>"),
    ("Could not resolve host", "DNS failure. Check: cat /etc/resolv.conf or try IP directly"),
    ("No space left on device", "Disk full. Check with: df -h"),
    ("Cannot allocate memory", "Out of memory. Check with: free -h"),
    ("ModuleNotFoundError", "Python module missing. Try: pip install <module>"),
    ("ImportError", "Python import failed. Check virtual environment: which python"),
    ("SyntaxError", "Code syntax error. Check the file at the line number shown"),
    ("ECONNREFUSED", "Connection refused. Service may not be running"),
    ("EACCES", "Permission denied. Check file/port permissions"),
    ("already in use", "Port already in use. Find process: lsof -i :<port>"),
    ("killed", "Process was killed (possibly OOM). Check: dmesg | tail"),
    ("npm ERR!", "npm error. Try: rm -rf node_modules && npm install"),
    ("E: Unable to locate package", "Package not found. Try: apt update first"),
]


def detect_errors(output: str) -> list[str]:
    """Detect common error patterns and return actionable hints."""
    hints: list[str] = []
    output_lower = output.lower()
    for pattern, hint in _ERROR_PATTERNS:
        if pattern.lower() in output_lower:
            hints.append(f"⚠ {hint}")
    return hints


def get_incremental(full_output: str, last_output: str) -> str:
    """Return only the new portion of output since last check."""
    if not last_output:
        return full_output
    if full_output.startswith(last_output):
        new = full_output[len(last_output):]
        return f"(incremental output)\n{new}" if new.strip() else "(no new output)"
    return full_output


def record_command(app: AppState, command: str, output: str, session: str) -> None:
    """Record command to history with error detection."""
    errors = detect_errors(output)
    record = CommandRecord(
        command=command, output=output[:500], session=session,
        success=not bool(errors), errors=errors,
    )
    history = app.get_history()
    history.append(record)
    if len(history) > 50:
        app.history = history[-50:]


def get_app(ctx: McpContext) -> AppState:  # type: ignore[type-arg]
    """Get AppState from MCP context and auto-initialize."""
    state = ctx.request_context.lifespan_context
    if not isinstance(state, AppState):
        raise RuntimeError("Server not initialized")
    if not state.initialized:
        state.initialized = True
        initialize("first_call", Context(state.bash_state, state.console),
                   "", [], "", CODING_MAX_TOKENS, NONCODING_MAX_TOKENS, "babash", "")
    return state


def get_app_from_request() -> AppState:
    """Get AppState from request context (for resources). Auto-initializes."""
    from mcp.server.lowlevel.server import request_ctx
    state = request_ctx.get().lifespan_context
    if not isinstance(state, AppState):
        raise RuntimeError("Server not initialized")
    if not state.initialized:
        state.initialized = True
        initialize("first_call", Context(state.bash_state, state.console),
                   "", [], "", CODING_MAX_TOKENS, NONCODING_MAX_TOKENS, "babash", "")
    return state


def ctx(app: AppState) -> Context:
    """Create tools Context from AppState."""
    return Context(app.bash_state, app.console)
