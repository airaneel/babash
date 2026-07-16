"""The FastMCP instance and its lifespan.

This lives apart from server.py so that tool and resource modules can import
`mcp` to decorate themselves without importing the server — which would import
them back. server.py is then free to be the module that pulls everything
together, rather than the one everything reaches into.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from importlib import metadata
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel.server import request_ctx
from mcp.types import ToolAnnotations
from pydantic import Field

from ...settings import Settings
from ..bash_state import get_tmpdir
from .state import AppState, Console

# Read once, from the environment, and never written again — so there is no
# question of what a shell's timings are or when they were set. Everything
# downstream gets them by argument, off AppState.
SETTINGS = Settings.from_env(get_tmpdir())

logging.basicConfig(
    level=logging.DEBUG if SETTINGS.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


INSTRUCTIONS = """babash is a terminal MCP server: persistent, isolated shells you can come back to.

# chat_id — REQUIRED, read first
One babash server is shared by many conversations. To keep your shell isolated
from other chats you MUST:
1. Call babash_initialize once at the very start of the conversation. It returns
   a chat_id.
2. Pass that same chat_id to EVERY babash tool call for the rest of the
   conversation (run_command, check_status, send_keys, read_file, … all of them).
Calls with an unknown/missing chat_id are rejected — they do not fall back to a
shared shell. Your chat_id maps to your own "main" shell plus your own named
sessions; other chats cannot see or disturb them.

# Shell commands
- run_command(command): execute a command and get output. Returns quickly. If the
  command is still running you get a "pending" status — that's normal, use
  check_status to poll or work in another session in the meantime.
- check_status(wait_for_seconds=N): get new output since the last check. N is capped at 5s server-side.
  For long-running commands (ansible, builds), space out your checks — read the incremental
  output each time and do other work in between. Don't call it in a tight loop expecting it to block.
- send_input(text): send text to a running interactive program (passwords, prompts).
- send_keys(keys): send special keys — "Ctrl-c" to interrupt, "Ctrl-d" for end-of-input (exits a
  REPL), "Enter" to confirm, Tab/Escape/arrows to drive a TUI.

# Interactive programs
If a command stops to ask something, the status says `waiting for input` and quotes the prompt —
answer it with send_input(text=...). Do NOT poll check_status at that point: the command is waiting
on you and nothing new will arrive until you answer.

# Sessions (parallel shells)
You start with a 'main' session. For parallel work, create named sessions:
- create_session(name="server") → independent shell
- run_command(command="npm start", session="server")
- check_status(session="server")
- send_keys(keys="Ctrl-c", session="server")
- destroy_session(name="server") → clean up when done

Use sessions when you need things running simultaneously:
  session "server" → npm start (keeps running)
  session "main"   → npm test (while server runs)

Use list_sessions() to see all sessions with their status and last command.

# Background commands
run_command(command="npm start", is_background=true) auto-creates a session (e.g. "bg_a1b2c3").
Use check_status(session="bg_a1b2c3") to monitor, send_keys("Ctrl-c", session="bg_a1b2c3") to stop.
Background sessions appear in list_sessions() and can be destroyed with destroy_session().

# File operations
- read_file(file_path): read a text file. Use offset/limit for large ones.
- read_document(file_path): read a PDF, Word, Excel or PowerPoint file as text.
- read_image(file_path): look at a PNG/JPEG/GIF/WebP — you see the actual image.
  Those four are the only formats the model can see; for HEIC/BMP/TIFF/SVG, or an
  image over 8000px, the tool tells you the one shell command that fixes it.
- write_file(file_path, content): create the file, or replace it whole.
- edit_file(file_path, old_string, new_string): replace an exact string. old_string
  must appear exactly once — include surrounding lines to make it unique — and must
  match the file byte for byte, so copy it out of a read_file. Pass replace_all=true
  to change every occurrence instead.

All file tools accept session= to work on files wherever that session's shell is —
including the far side of an SSH connection:
  read_file(file_path="/etc/hosts", session="myserver")
  write_file(file_path="/etc/config.yaml", content="...", session="myserver")

Do NOT use echo/cat/sed/heredocs to read or write files — use the file tools. They
move content as base64, so quotes, $, and backticks in YAML/JSON/Jinja survive
verbatim instead of being mangled by the shell's parser.

# Important
- Each session runs one foreground command at a time.
- If a command is still running, check_status or send_keys(Ctrl-c) before running another.
- Do NOT poll check_status repeatedly waiting for a command to finish. If the command has no output
  after one check, either send_keys(Ctrl-c) and try a different approach, or move on to other work
  in a different session. Never call check_status more than 2-3 times for the same command.
- cd, env vars, and state persist within each session independently.
- If output is truncated, use more precise commands (grep, head, tail, awk) instead of dumping everything.
- For large files, use read_file with offset/limit instead of reading the whole thing.
- For SSH: open an interactive session with run_command("ssh user@host"), then run commands directly.
  Do NOT use ssh user@host "cmd" repeatedly — it reconnects and re-authenticates every time.
  Use a session for long SSH work: create_session("remote"), run_command("ssh user@host", session="remote").
"""


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppState]:
    """Hold the server's state for as long as it runs, and tear it down after.

    No shell is created here: every shell belongs to a chat_id and is spawned
    lazily by babash_initialize, so that two conversations sharing this process
    never share a pty. What must happen on the way out is that they are all
    killed — a leaked pty outlives the server.
    """
    console = Console()
    console.log("babash version: " + str(metadata.version("babash")))

    app = AppState(settings=SETTINGS, console=console, chats={})
    try:
        yield app
    finally:
        app.cleanup()


mcp = FastMCP(
    "babash",
    instructions=INSTRUCTIONS,
    lifespan=app_lifespan,
    host=SETTINGS.host,
    port=SETTINGS.port,
)


# The two parameters nearly every tool takes. Declared once, so the model gets
# the same account of what they mean on every tool rather than the bare
# `{"title": "Chat Id", "type": "string"}` that an unannotated `str` compiles
# down to — which says nothing about where the value comes from or why it
# matters. The description is prompt surface: it ships to the model with
# tools/list, and it is the only chance to explain a parameter *before* the
# model guesses at it.
ChatId = Annotated[
    str,
    Field(
        description=(
            "The chat_id babash_initialize returned for this conversation. It is "
            "what keeps this chat's shells separate from every other chat's on "
            "the same server — pass the same one on every call."
        ),
    ),
]

Session = Annotated[
    str | None,
    Field(
        description=(
            "Which of this chat's shells to act in. Omit for the default 'main' "
            "shell; pass a name from create_session (or list_sessions) to work in "
            "a parallel one — including one sitting inside an SSH connection."
        ),
    ),
]


def text_tool(
    description: str,
    annotations: ToolAnnotations,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a tool whose result is prose for the model to read.

    Every babash tool returns human-readable text — command output, a status
    line, an error with the shell command that fixes it — never a JSON object a
    client parses against a schema. The MCP spec calls that unstructured content,
    and it belongs in the reply's `content` as a single TextContent block.

    Left to its default, FastMCP does something else. It sees the `-> str` return
    annotation, decides the tool is "structured", auto-builds an output schema of
    `{result: string}`, and returns the reply twice: as the text we wrote *and*
    as `structuredContent={"result": "<the same text>"}`. Claude Desktop, handed
    both, renders the structured half — so the user reads `{"result": "..."}`
    instead of the output. There is nothing structured about a string; the schema
    is noise. structured_output=False drops it, and the reply is just the text.

    (It also spares read_image, whose return is an Image the schema builder cannot
    describe at all — with the default it would fail to register outright.)
    """
    return mcp.tool(
        description=description,
        annotations=annotations,
        structured_output=False,
    )


def get_app() -> AppState:
    """This request's AppState.

    Read from the contextvar rather than from a `Context` parameter, because
    that is where it actually lives: the lowlevel server sets `request_ctx` once
    per request, for tools and resources alike, and FastMCP's `Context` object
    is only a handle onto the same RequestContext. Going to the source means one
    accessor instead of one per call style — and it means a tool only has to
    declare a `ctx` parameter when it genuinely uses one, to report progress or
    log, rather than as a token to exchange for this.
    """
    state = request_ctx.get().lifespan_context
    if not isinstance(state, AppState):
        raise RuntimeError("Server not initialized")
    return state
