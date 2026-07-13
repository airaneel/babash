"""Tests for MCP server via subprocess (end-to-end protocol tests)."""

import json
import subprocess
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
    r = _call(send, recv, "babash_initialize", {"type": "first_call"}, req_id)
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
        assert "read_files_tool" in names
        assert "file_write_or_edit" in names
        assert "context_save" in names
        assert "babash_initialize" in names
    finally:
        proc.terminate()


def test_list_prompts() -> None:
    proc, send, recv, _ = _session()
    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "prompts/list", "params": {}})
        r = recv()
        prompts = r["result"]["prompts"]
        assert len(prompts) > 0
        assert any(p["name"] == "KnowledgeTransfer" for p in prompts)
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


def test_babash_initialize() -> None:
    proc, send, recv, _ = _session()
    try:
        r = _call(send, recv, "babash_initialize", {"type": "first_call"}, 1)
        text = r["result"]["content"][0]["text"]
        assert "Initialize call done" in text
        assert "Your chat_id is:" in text
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
