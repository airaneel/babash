"""MCP resources and the health route.

Importing this module registers them on the shared `mcp` instance.
"""

import platform
import shutil
from typing import Any

from starlette.responses import JSONResponse

from .instance import get_app, mcp


@mcp.resource("babash://workspace/env", description="Shell environment and system info")
def workspace_env() -> str:
    app = get_app()
    lines = [
        f"system: {platform.system()} {platform.release()}",
        f"machine: {platform.machine()}",
        f"active_chats: {len(app.chats)}",
    ]
    for chat in app.chats.values():
        bs = chat.main
        lines.append(f"chat {chat.chat_id}: cwd={bs.cwd} state={bs.state}")
    for tool in ["git", "docker", "python3", "node", "npm", "uv", "pip", "rg", "jq", "ssh", "curl"]:
        path = shutil.which(tool)
        if path:
            lines.append(f"has_{tool}: {path}")
    return "\n".join(lines)


@mcp.resource(
    "babash://workspace/processes",
    description="All chats, sessions and running commands",
)
def workspace_processes() -> str:
    app = get_app()
    if not app.chats:
        return "(no chats initialized yet)"
    lines: list[str] = []
    for chat in app.chats.values():
        for name, shell in [("main", chat.main), *chat.sessions.items()]:
            lines.append(
                f"[chat {chat.chat_id}] {name}: cwd={shell.cwd} state={shell.state} "
                f"cmd={shell.last_command or '(idle)'}"
            )
    return "\n".join(lines)


@mcp.resource(
    "babash://history",
    description="Command history with success/failure and error hints",
)
def command_history() -> str:
    app = get_app()
    records = [(chat.chat_id, rec) for chat in app.chats.values() for rec in chat.history]
    if not records:
        return "No commands executed yet."
    lines = []
    for i, (cid, rec) in enumerate(records[-20:], 1):
        status = "✓" if rec.success else "✗"
        lines.append(f"{i}. [{status}] [chat {cid}] [{rec.session}] $ {rec.command}")
        for err in rec.errors:
            lines.append(f"   {err}")
    return "\n".join(lines)


@mcp.custom_route("/health", methods=["GET"])  # type: ignore[untyped-decorator]
async def health_check(request: Any) -> Any:
    return JSONResponse({"status": "ok", "server": "babash"})
