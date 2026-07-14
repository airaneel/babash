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

from mcp.server.fastmcp.utilities.types import Image
from mcp.types import ToolAnnotations

from ...documents import DocumentError, extract
from ...fs import FileError, FileStore, LocalStore, SessionStore
from ...images import ImageError, load
from ..chat import resolve_chat
from ..instance import get_app, mcp
from ..state import ChatWorkspace

# Reading a whole file at once is usually a mistake on anything large; this is
# where we make the agent say what it actually wants.
DEFAULT_READ_LIMIT = 2000

def _store(chat: ChatWorkspace, session: str | None) -> FileStore | str:
    """Where this call's files live, or an error to hand back."""
    if session is None:
        return LocalStore()
    try:
        return SessionStore(chat.get_shell(session), session)
    except ValueError as e:
        return f"Error: {e}"


def _numbered(content: str, offset: int, limit: int) -> str:
    """A slice of the file with line numbers, cat -n style."""
    lines = content.splitlines()
    start = max(1, offset)
    chunk = lines[start - 1 : start - 1 + limit]
    body = "\n".join(f"{start + i:6d}\t{line}" for i, line in enumerate(chunk))

    shown_to = start + len(chunk) - 1
    if len(lines) > shown_to:
        body += (
            f"\n\n(showing lines {start}-{shown_to} of {len(lines)}; "
            f"pass offset={shown_to + 1} to continue)"
        )
    return body or "(empty file)"


@mcp.tool(
    description=(
        "Read a file. Returns numbered lines. Use offset/limit for large files. "
        "Pass session= to read a file on the far side of that session's shell "
        "(e.g. one sitting in an SSH connection)."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def read_file(
    file_path: str,
    chat_id: str,
    session: str | None = None,
    offset: int = 1,
    limit: int = DEFAULT_READ_LIMIT,
) -> str:
    app = get_app()
    chat, err = resolve_chat(app, chat_id)
    if chat is None:
        return err
    store = _store(chat, session)
    if isinstance(store, str):
        return store
    try:
        content = store.read(file_path)
    except FileError as e:
        return f"Error: {e}"
    return _numbered(content, offset, limit)[:app.settings.max_output_chars]


@mcp.tool(
    description=(
        "Look at an image — a screenshot, a diagram, a rendered plot. PNG, JPEG, GIF "
        "and WebP are the formats the model can see; for anything else (HEIC, BMP, "
        "TIFF, SVG) this tells you the one shell command that converts it. Pass "
        "session= to fetch an image from the far side of that session's shell, e.g. "
        "a screenshot taken on a remote host."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    # This tool returns an image, not JSON. FastMCP would otherwise try to build
    # an output schema out of the return annotation, and pydantic cannot describe
    # an Image; the result goes back as ImageContent, which needs no schema.
    structured_output=False,
)
async def read_image(
    file_path: str,
    chat_id: str,
    session: str | None = None,
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
        image = load(store.read_bytes(file_path))
    except FileError as e:
        return f"Error: {e}"
    except ImageError as e:
        # Not "Error:" — these say which shell command turns the file into
        # something the model can see, which the agent can run itself.
        return str(e)

    return Image(data=image.data, format=image.format)


@mcp.tool(
    description=(
        "Read a PDF, Word, Excel or PowerPoint file as text. Use this instead of "
        "read_file for anything that isn't plain text. Pass session= to read one on "
        "the far side of that session's shell."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def read_document(
    file_path: str,
    chat_id: str,
    session: str | None = None,
) -> str:
    app = get_app()
    chat, err = resolve_chat(app, chat_id)
    if chat is None:
        return err
    store = _store(chat, session)
    if isinstance(store, str):
        return store

    try:
        doc = extract(store.read_bytes(file_path))
    except FileError as e:
        return f"Error: {e}"
    except DocumentError as e:
        # Not "Error:" — these say what to do next, and most of them are asking
        # the agent to convert the file in the shell it already has.
        return str(e)

    header = f"<{doc.kind} path=\"{file_path}\">"
    body = doc.note if doc.note else doc.text
    return f"{header}\n{body}\n</{doc.kind}>"[: app.settings.max_output_chars]


@mcp.tool(
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
        openWorldHint=False,
    ),
)
async def write_file(
    file_path: str,
    content: str,
    chat_id: str,
    session: str | None = None,
) -> str:
    app = get_app()
    chat, err = resolve_chat(app, chat_id)
    if chat is None:
        return err
    store = _store(chat, session)
    if isinstance(store, str):
        return store

    existed = store.exists(file_path)
    try:
        store.write(file_path, content)
    except FileError as e:
        return f"Error: {e}"

    verb = "Overwrote" if existed else "Created"
    lines = content.count("\n") + 1 if content else 0
    return f"{verb} {file_path} {store.where} ({lines} lines, {len(content.encode())} bytes)."


@mcp.tool(
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
        openWorldHint=False,
    ),
)
async def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    chat_id: str,
    session: str | None = None,
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
        content = store.read(file_path)
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
        store.write(file_path, content.replace(old_string, new_string))
    except FileError as e:
        return f"Error: {e}"

    where = f" {store.where}" if session else ""
    plural = "s" if count > 1 else ""
    return f"Replaced {count} occurrence{plural} in {file_path}{where}."
