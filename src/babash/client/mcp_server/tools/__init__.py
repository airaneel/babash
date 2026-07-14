"""The MCP tools babash exposes.

Each module here decorates its functions with `@mcp.tool` against the shared
instance in ../instance.py, so importing the module is what registers them.
`__all__` names them so that import is not mistaken for an unused one.
"""

from .files import edit_file, read_document, read_file, read_image, write_file
from .sessions import create_session, destroy_session, list_sessions
from .shell import babash_initialize, check_status, run_command, send_input, send_keys

__all__ = [
    "babash_initialize",
    "check_status",
    "create_session",
    "destroy_session",
    "edit_file",
    "list_sessions",
    "read_document",
    "read_file",
    "read_image",
    "run_command",
    "send_input",
    "send_keys",
    "write_file",
]
