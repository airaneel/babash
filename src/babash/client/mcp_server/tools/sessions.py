"""Creating, listing and tearing down a chat's named shells."""

import pexpect
from mcp.types import ToolAnnotations

from ..chat import full_roster, resolve_chat, warmup_shell
from ..instance import ChatId, get_app, text_tool


@text_tool(
    description="Create a named shell session for parallel work.",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
async def create_session(
    name: str,
    chat_id: ChatId,
    working_directory: str = "",
) -> str:
    app = get_app()
    chat, err = resolve_chat(app, chat_id)
    if chat is None:
        return err
    if name == "main":
        return "Error: 'main' already exists."
    if name in chat.sessions:
        return f"Session '{name}' already exists."
    cwd = working_directory or chat.main.cwd
    chat.sessions[name] = app.new_shell(cwd)
    await warmup_shell(chat.sessions[name])
    return f"Session '{name}' created (cwd: {cwd}).\n\n{full_roster(chat)}"


@text_tool(
    description="List all shell sessions and their status.",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        openWorldHint=False,
    ),
)
async def list_sessions(chat_id: ChatId) -> str:
    app = get_app()
    chat, err = resolve_chat(app, chat_id)
    if chat is None:
        return err
    return full_roster(chat)


@text_tool(
    description="Destroy a named session.",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def destroy_session(name: str, chat_id: ChatId) -> str:
    app = get_app()
    chat, err = resolve_chat(app, chat_id)
    if chat is None:
        return err
    if name == "main":
        return "Error: cannot destroy main session."
    if name not in chat.sessions:
        return f"Session '{name}' not found."
    # Popped before it is killed: the session is gone from the agent's point of
    # view whether or not the pty goes quietly.
    shell = chat.sessions.pop(name)
    try:
        shell.sendintr()
        shell.cleanup()
    except (pexpect.ExceptionPexpect, OSError) as e:
        return f"Session '{name}' destroyed, but its shell did not close cleanly: {e}\n\n{full_roster(chat)}"
    return f"Session '{name}' destroyed.\n\n{full_roster(chat)}"
