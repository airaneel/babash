"""Output post-processing."""

from .state import ChatWorkspace, CommandRecord

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
    """Common failure modes, and what to do about them."""
    output_lower = output.lower()
    return [
        f"⚠ {hint}"
        for pattern, hint in _ERROR_PATTERNS
        if pattern.lower() in output_lower
    ]


def record_command(chat: ChatWorkspace, command: str, output: str, session: str) -> None:
    """Add a command to a chat's history, noting whether it looks like it failed."""
    errors = detect_errors(output)
    chat.history.append(
        CommandRecord(
            command=command,
            output=output[:500],
            session=session,
            success=not bool(errors),
            errors=errors,
        )
    )
    if len(chat.history) > 50:
        del chat.history[:-50]
