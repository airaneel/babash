"""Tests for MCP server via subprocess (end-to-end protocol tests)."""

import json
import subprocess
import time
import sys
from typing import Any, Callable

BABASH_MCP = [sys.executable, "-m", "babash.client.mcp_server"]

SendFn = Callable[[dict[str, Any]], None]
RecvFn = Callable[[], dict[str, Any]]


def _session() -> tuple[subprocess.Popen[str], SendFn, RecvFn, dict[str, Any]]:
    """Start babash_mcp and return (proc, send, recv, init_result)."""
    proc = subprocess.Popen(
        BABASH_MCP,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def send(msg: dict[str, Any]) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

    def recv() -> dict[str, Any]:
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("Server closed")
            msg: dict[str, Any] = json.loads(line)
            if "id" in msg:
                return msg

    send(
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        }
    )
    init_result = recv()
    send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    return proc, send, recv, init_result


def _call(send: SendFn, recv: RecvFn, name: str, arguments: dict[str, Any], req_id: int) -> dict[str, Any]:
    send(
        {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    return recv()


def _init_chat(send: SendFn, recv: RecvFn, req_id: int) -> str:
    """Call babash_initialize and return the assigned chat_id."""
    r = _call(send, recv, "babash_initialize", {}, req_id)
    text: str = r["result"]["content"][0]["text"]
    for line in text.splitlines():
        if line.startswith("Your chat_id is:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError("no chat_id in init output:\n" + text)


def test_server_init() -> None:
    proc, send, recv, init_result = _session()
    try:
        assert init_result["result"]["serverInfo"]["name"] == "babash"
        assert "instructions" in init_result["result"]
        assert len(init_result["result"]["instructions"]) > 0
    finally:
        proc.terminate()


def test_list_tools() -> None:
    proc, send, recv, _ = _session()
    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        r = recv()
        tools = r["result"]["tools"]
        names = {t["name"] for t in tools}
        assert "run_command" in names
        assert "check_status" in names
        assert "send_input" in names
        assert "send_keys" in names
        assert "read_file" in names
        assert "write_file" in names
        assert "edit_file" in names
        assert "create_session" in names
        assert "babash_initialize" in names
    finally:
        proc.terminate()


def test_run_command() -> None:
    proc, send, recv, _ = _session()
    try:
        cid = _init_chat(send, recv, 1)
        r = _call(send, recv, "run_command", {"command": "echo mcp-test-pass", "chat_id": cid}, 2)
        text = r["result"]["content"][0]["text"]
        assert "mcp-test-pass" in text
    finally:
        proc.terminate()


def test_check_status() -> None:
    proc, send, recv, _ = _session()
    try:
        cid = _init_chat(send, recv, 1)
        r = _call(send, recv, "check_status", {"chat_id": cid}, 2)
        assert "result" in r
    finally:
        proc.terminate()


def test_polling_a_slow_command_stays_small() -> None:
    """An agent waiting on a build polls every few seconds. Each of those replies
    used to restate the running command twice — once in the status line, once in
    the roster — and list every idle shell besides. Nothing in it changed between
    polls, so it was the same several hundred tokens over and over."""
    proc, send, recv, _ = _session()
    try:
        cid = _init_chat(send, recv, 1)
        _call(send, recv, "create_session", {"name": "watch", "chat_id": cid}, 2)

        command = "echo begin; sleep 20; echo end"
        _call(send, recv, "run_command", {"command": command, "chat_id": cid}, 3)

        poll = _call(send, recv, "check_status", {"chat_id": cid, "wait_for_seconds": 2}, 4)
        text = poll["result"]["content"][0]["text"]

        assert "still running" in text
        assert "watch" not in text          # idle: not news, not repeated
        assert text.count(command) <= 1     # said once, not twice
        assert len(text) < 300

        # And the roster is still there for the asking.
        listed = _call(send, recv, "list_sessions", {"chat_id": cid}, 5)
        assert "watch: idle" in listed["result"]["content"][0]["text"]
    finally:
        proc.terminate()


def test_babash_initialize() -> None:
    proc, send, recv, _ = _session()
    try:
        r = _call(send, recv, "babash_initialize", {}, 1)
        text = r["result"]["content"][0]["text"]
        assert "Your chat_id is:" in text
    finally:
        proc.terminate()


def test_file_tools_roundtrip(tmp_path: Any) -> None:
    """write -> read -> edit, over the wire."""
    proc, send, recv, _ = _session()
    try:
        cid = _init_chat(send, recv, 1)
        path = str(tmp_path / "note.txt")

        r = _call(send, recv, "write_file",
                  {"file_path": path, "content": "alpha\nbeta\n", "chat_id": cid}, 2)
        assert "Created" in r["result"]["content"][0]["text"]

        r = _call(send, recv, "read_file", {"file_path": path, "chat_id": cid}, 3)
        assert "alpha" in r["result"]["content"][0]["text"]

        r = _call(send, recv, "edit_file",
                  {"file_path": path, "old_string": "beta", "new_string": "GAMMA",
                   "chat_id": cid}, 4)
        assert "Replaced 1 occurrence" in r["result"]["content"][0]["text"]

        r = _call(send, recv, "read_file", {"file_path": path, "chat_id": cid}, 5)
        text = r["result"]["content"][0]["text"]
        assert "GAMMA" in text and "beta" not in text
    finally:
        proc.terminate()


def test_sessions_are_independent(tmp_path: Any) -> None:
    proc, send, recv, _ = _session()
    try:
        cid = _init_chat(send, recv, 1)
        _call(send, recv, "create_session", {"name": "other", "chat_id": cid}, 2)

        _call(send, recv, "run_command",
              {"command": "export WHO=main", "chat_id": cid}, 3)
        _call(send, recv, "run_command",
              {"command": "export WHO=other", "chat_id": cid, "session": "other"}, 4)

        r = _call(send, recv, "run_command", {"command": "echo [$WHO]", "chat_id": cid}, 5)
        assert "[main]" in r["result"]["content"][0]["text"]

        r = _call(send, recv, "run_command",
                  {"command": "echo [$WHO]", "chat_id": cid, "session": "other"}, 6)
        assert "[other]" in r["result"]["content"][0]["text"]
    finally:
        proc.terminate()


def test_run_command_requires_chat_id() -> None:
    """Isolation contract: an unknown chat_id is rejected (no silent shared
    shell), and the chat_id returned by babash_initialize works."""
    proc, send, recv, _ = _session()
    try:
        r = _call(send, recv, "run_command", {"command": "echo hi", "chat_id": "does-not-exist"}, 1)
        text = r["result"]["content"][0]["text"]
        assert "unknown chat_id" in text

        cid = _init_chat(send, recv, 2)
        r2 = _call(send, recv, "run_command", {"command": "echo ok-isolated", "chat_id": cid}, 3)
        assert "ok-isolated" in r2["result"]["content"][0]["text"]
    finally:
        proc.terminate()


def test_background_returns_before_the_command_finishes() -> None:
    """is_background is the agent saying it will not wait — so don't make it."""
    proc, send, recv, _ = _session()
    try:
        cid = _init_chat(send, recv, 1)
        started = time.monotonic()
        r = _call(send, recv, "run_command",
                  {"command": "sleep 30", "chat_id": cid, "is_background": True}, 2)
        elapsed = time.monotonic() - started
        assert "Running in session 'bg_" in r["result"]["content"][0]["text"]
        assert elapsed < 8, f"background launch took {elapsed:.1f}s"
    finally:
        proc.terminate()


def test_background_still_reports_an_immediate_failure() -> None:
    """The short budget must not cost the agent the news that the command died."""
    proc, send, recv, _ = _session()
    try:
        cid = _init_chat(send, recv, 1)
        r = _call(send, recv, "run_command",
                  {"command": "nosuchbinary --go", "chat_id": cid, "is_background": True}, 2)
        text = r["result"]["content"][0]["text"]
        assert "not found" in text
        assert "exit code = 127" in text
    finally:
        proc.terminate()


def test_duplicate_background_commands_run_side_by_side() -> None:
    """Same command, same cwd, twice: the agent wants two workers, not a refusal.

    The session name is derived from cwd+command so a re-run lands back in the
    same shell rather than leaking a new one — but only if that shell is free.
    """
    proc, send, recv, _ = _session()
    try:
        cid = _init_chat(send, recv, 1)
        names = []
        for i in (2, 3):
            r = _call(send, recv, "run_command",
                      {"command": "sleep 20", "chat_id": cid, "is_background": True}, i)
            text = r["result"]["content"][0]["text"]
            names.append(text.split("Running in session '")[1].split("'")[0])

        assert names[0] != names[1], "a busy bg session must not be reused"

        roster = _call(send, recv, "list_sessions", {"chat_id": cid}, 4)
        roster_text = roster["result"]["content"][0]["text"]
        for name in names:
            assert f"{name}: running" in roster_text, f"{name} should be running"
    finally:
        proc.terminate()
