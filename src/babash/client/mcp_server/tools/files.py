"""File tools: read, write, edit — locally or through a session's shell.

Editing is by exact string match. What this replaced was upstream's design: the
agent sent SEARCH/REPLACE blocks plus a guess at what percentage of the file it
was about to change, and ~1600 lines then tried to make that land — fuzzy
matching when the search text didn't quite match, re-indenting the replacement
to fit, and a whitelist recording which line ranges of which files had been read
so an overwrite could be refused if the agent hadn't looked first.

Exact matching dissolves all three. A match either is or isn't the text the
agent asked for, so there is nothing to be tolerant about; the replacement goes
in verbatim, so there is nothing to re-indent; and quoting the existing text
back at us *is* proof of having read it, so there is nothing to whitelist. When
it can't be done, we say so and the agent reads the file — which is what the
whitelist was trying to force anyway.
"""

from typing import Annotated

import anyio
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import ToolAnnotations
from pydantic import Field

from ...documents import DocumentError, extract
from ...fs import FileError, FileStore, LocalStore, SessionStore
from ...images import ImageError, load
from ..chat import resolve_chat
from ..instance import ChatId, Session, get_app, text_tool
from ..state import ChatWorkspace

# Reading a whole file at once is usually a mistake on anything large; this is
# where we make the agent say what it actually wants.
DEFAULT_READ_LIMIT = 2000

FilePath = Annotated[
    str,
    Field(
        description=(
            "Path to the file. Absolute, or relative to the working directory of "
            "the shell this call runs against (see session)."
        ),
    ),
]

def _store(chat: ChatWorkspace, session: str | None) -> FileStore | str:
    """Where this call's files live, or an error to hand back."""
    if session is None:
        return LocalStore()
    try:
        return SessionStore(chat.get_shell(session), session)
    except ValueError as e:
        return f"Error: {e}"


# Every store call below goes through a worker thread. A FileStore is
# deliberately synchronous — LocalStore opens a file, SessionStore sends
# `base64 < path` down a pty and waits for the prompt — and calling either
# straight from an `async def` parks the whole event loop on it. That is not
# theoretical: a single read_file against a *local* session measured 1304ms with
# zero heartbeat ticks getting through, and a session sitting in an SSH
# connection turns each of those round trips into a network one. One chat
# reading a file would stall every other chat's tool call on the shared server.
#
# shell.py already hands execute_bash to anyio.to_thread for exactly this
# reason; the file tools simply never followed.


def _numbered(content: str, offset: int, limit: int) -> str:
    """A slice of the file with line numbers, cat -n style.

    `offset` is 1-based and `limit` is positive — the schema guarantees both, so
    there is no clamping here. It used to clamp `max(1, offset)`, which quietly
    turned a nonsense offset into a read of the top of the file.
    """
    lines = content.splitlines()
    chunk = lines[offset - 1 : offset - 1 + limit]

    if not chunk:
        # Distinguish the two ways a read comes back empty. This used to answer
        # "(empty file)" to both — so a read past the end of a perfectly good
        # file told the model the file had nothing in it.
        if not lines:
            return "(empty file)"
        return (
            f"(no lines at offset {offset}: the file has {len(lines)} lines. "
            f"Pass an offset of {len(lines)} or less.)"
        )

    body = "\n".join(f"{offset + i:6d}\t{line}" for i, line in enumerate(chunk))
    shown_to = offset + len(chunk) - 1
    if len(lines) > shown_to:
        body += (
            f"\n\n(showing lines {offset}-{shown_to} of {len(lines)}; "
            f"pass offset={shown_to + 1} to continue)"
        )
    return body


@text_tool(
    description=(
        "Read a file. Returns numbered lines. Use offset/limit for large files. "
        "Pass session= to read a file on the far side of that session's shell "
        "(e.g. one sitting in an SSH connection)."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        openWorldHint=True,
    ),
)
async def read_file(
    file_path: FilePath,
    chat_id: ChatId,
    session: Session = None,
    offset: Annotated[
        int,
        Field(ge=1, description="First line to return, counting from 1."),
    ] = 1,
    limit: Annotated[
        int,
        Field(ge=1, description="How many lines to return, starting at offset."),
    ] = DEFAULT_READ_LIMIT,
) -> str:
    app = get_app()
    chat, err = resolve_chat(app, chat_id)
    if chat is None:
        return err
    store = _store(chat, session)
    if isinstance(store, str):
        return store
    try:
        content = await anyio.to_thread.run_sync(store.read, file_path)
    except FileError as e:
        return f"Error: {e}"
    return _numbered(content, offset, limit)[:app.settings.max_output_chars]


@text_tool(
    description=(
        "Look at an image — a screenshot, a diagram, a rendered plot. PNG, JPEG, GIF "
        "and WebP are the formats the model can see; for anything else (HEIC, BMP, "
        "TIFF, SVG) this tells you the one shell command that converts it. Pass "
        "session= to fetch an image from the far side of that session's shell, e.g. "
        "a screenshot taken on a remote host."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        openWorldHint=True,
    ),
)
async def read_image(
    file_path: FilePath,
    chat_id: ChatId,
    session: Session = None,
) -> Image | str:
    # Returns a real Image, which FastMCP turns into ImageContent — so the model
    # actually sees the picture. The version of this tool that used to exist read
    # the file, base64'd it, and then returned the string
    # "[Image: image/png, 12345 bytes base64]" — a description of an image the
    # model was never shown.
    app = get_app()
    chat, err = resolve_chat(app, chat_id)
    if chat is None:
        return err
    store = _store(chat, session)
    if isinstance(store, str):
        return store

    try:
        data = await anyio.to_thread.run_sync(store.read_bytes, file_path)
        image = load(data)
    except FileError as e:
        return f"Error: {e}"
    except ImageError as e:
        # Not "Error:" — these say which shell command turns the file into
        # something the model can see, which the agent can run itself.
        return str(e)

    return Image(data=image.data, format=image.format)


@text_tool(
    description=(
        "Read a PDF, Word, Excel or PowerPoint file as text. Use this instead of "
        "read_file for anything that isn't plain text. Pass session= to read one on "
        "the far side of that session's shell."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        openWorldHint=True,
    ),
)
async def read_document(
    file_path: FilePath,
    chat_id: ChatId,
    session: Session = None,
) -> str:
    app = get_app()
    chat, err = resolve_chat(app, chat_id)
    if chat is None:
        return err
    store = _store(chat, session)
    if isinstance(store, str):
        return store

    try:
        data = await anyio.to_thread.run_sync(store.read_bytes, file_path)
        # extract() too: parsing a few hundred PDF pages is seconds of CPU, and
        # the loop should not be sitting inside pypdf either.
        doc = await anyio.to_thread.run_sync(extract, data)
    except FileError as e:
        return f"Error: {e}"
    except DocumentError as e:
        # Not "Error:" — these say what to do next, and most of them are asking
        # the agent to convert the file in the shell it already has.
        return str(e)

    header = f"<{doc.kind} path=\"{file_path}\">"
    body = doc.note if doc.note else doc.text
    return f"{header}\n{body}\n</{doc.kind}>"[: app.settings.max_output_chars]


@text_tool(
    description=(
        "Write a file, creating it or replacing it whole. Parent directories are "
        "created. Prefer this over `cat <<EOF` in run_command: the content never "
        "goes through a shell parser, so quotes, $, and backticks in YAML/JSON/"
        "Jinja survive verbatim. Pass session= to write on the far side of that "
        "session's shell. To change part of a file, use edit_file instead."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def write_file(
    file_path: FilePath,
    content: str,
    chat_id: ChatId,
    session: Session = None,
) -> str:
    app = get_app()
    chat, err = resolve_chat(app, chat_id)
    if chat is None:
        return err
    store = _store(chat, session)
    if isinstance(store, str):
        return store

    try:
        # exists() belongs inside the try: on a SessionStore it is a pty round
        # trip like any other, and it raises FileError when the session is busy.
        # Outside, that escaped as an unhandled exception rather than the message
        # telling the agent to check_status or Ctrl-c first.
        existed = await anyio.to_thread.run_sync(store.exists, file_path)
        await anyio.to_thread.run_sync(store.write, file_path, content)
    except FileError as e:
        return f"Error: {e}"

    verb = "Overwrote" if existed else "Created"
    lines = content.count("\n") + 1 if content else 0
    return f"{verb} {file_path} {store.where} ({lines} lines, {len(content.encode())} bytes)."


@text_tool(
    description=(
        "Replace an exact string in a file. old_string must appear EXACTLY once "
        "(include surrounding lines to make it unique), unless replace_all=true. "
        "Copy old_string verbatim from a read_file — whitespace and indentation "
        "must match. To delete text, pass an empty new_string. Pass session= to "
        "edit on the far side of that session's shell."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def edit_file(
    file_path: FilePath,
    old_string: str,
    new_string: str,
    chat_id: ChatId,
    session: Session = None,
    replace_all: bool = False,
) -> str:
    app = get_app()
    chat, err = resolve_chat(app, chat_id)
    if chat is None:
        return err
    store = _store(chat, session)
    if isinstance(store, str):
        return store

    if not old_string:
        return "Error: old_string cannot be empty. To create a file, use write_file."
    if old_string == new_string:
        return "Error: old_string and new_string are identical; nothing to do."

    try:
        content = await anyio.to_thread.run_sync(store.read, file_path)
    except FileError as e:
        return f"Error: {e}"

    count = content.count(old_string)
    if count == 0:
        return (
            f"Error: old_string not found in {file_path}. It must match the file "
            f"exactly, including whitespace and indentation — read the file and "
            f"copy the text verbatim."
        )
    if count > 1 and not replace_all:
        return (
            f"Error: old_string appears {count} times in {file_path}. Add "
            f"surrounding lines to make it unique, or pass replace_all=true to "
            f"replace every occurrence."
        )

    try:
        await anyio.to_thread.run_sync(
            store.write, file_path, content.replace(old_string, new_string)
        )
    except FileError as e:
        return f"Error: {e}"

    where = f" {store.where}" if session else ""
    plural = "s" if count > 1 else ""
    return f"Replaced {count} occurrence{plural} in {file_path}{where}."
