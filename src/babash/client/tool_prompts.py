import os

from mcp.types import Tool, ToolAnnotations

from ..types_ import (
    BashCommand,
    ContextSave,
    FileWriteOrEdit,
    Initialize,
    ReadFiles,
    ReadImage,
)
from .schema_generator import remove_titles_from_schema

with open(os.path.join(os.path.dirname(__file__), "diff-instructions.txt")) as f:
    diffinstructions = f.read()


TOOL_PROMPTS = [
    Tool(
        inputSchema=remove_titles_from_schema(Initialize.model_json_schema()),
        name="Initialize",
        description="""Initialize the shell environment. Must be called first before any other tool.
- Set `any_workspace_path` to the project directory. Use empty string if unknown.
- Set `initial_files_to_read` to files the user mentioned, or [] if none.
- Set `task_id_to_resume` to resume a previous task, or empty string for new tasks.
- Set `mode_name` to "babash" (full access, default), "architect" (read-only), or "code_writer" (restricted).
- Set `thread_id` to empty string on first_call. Use the returned thread_id for all subsequent tool calls.
- Set `type` to "first_call" for initial setup, "user_asked_mode_change" to switch modes, "reset_shell" if shell is broken, "user_asked_change_workspace" to change directory.
""",
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    ),
    Tool(
        inputSchema=remove_titles_from_schema(BashCommand.model_json_schema()),
        name="BashCommand",
        description="""Execute shell commands or interact with running processes.
IMPORTANT: Each `type` value has its own required field. Do NOT mix them.
- type="command" → set `command` (string). Optionally set `is_background` for background execution.
- type="status_check" → set `status_check` to true. No other fields needed.
- type="send_text" → set `send_text` (string). Do NOT use `command` field.
- type="send_specials" → set `send_specials` (array of keys like "Enter", "Ctrl-c", "Key-up").
- type="send_ascii" → set `send_ascii` (array of integer ASCII codes).
Always set `thread_id` to the value returned by Initialize.
For background commands: set `is_background` to true with type="command".
To interact with a background command: set `bg_command_id` on non-command types.
Only one foreground command runs at a time. Check status before running a new one.
Do not use echo/cat to read/write files — use ReadFiles/FileWriteOrEdit instead.
""",
        annotations=ToolAnnotations(destructiveHint=True, openWorldHint=True),
    ),
    Tool(
        inputSchema=remove_titles_from_schema(ReadFiles.model_json_schema()),
        name="ReadFiles",
        description="""Read content of one or more files.
- Provide absolute paths only (~ allowed).
- Supports line ranges: `/path/file.py:10-20` for lines 10-20, `:10-` from line 10, `:-20` first 20 lines.
""",
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    ),
    Tool(
        inputSchema=remove_titles_from_schema(ReadImage.model_json_schema()),
        name="ReadImage",
        description="Read an image file and return its contents. Provide absolute path.",
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    ),
    Tool(
        inputSchema=remove_titles_from_schema(FileWriteOrEdit.model_json_schema()),
        name="FileWriteOrEdit",
        description="""Write or edit a file.
- Set `thread_id` to the value returned by Initialize.
- Set `percentage_to_change`: estimate what % of existing lines will change (0-100).
- If percentage_to_change > 50: provide full file content in `text_or_search_replace_blocks`.
- If percentage_to_change <= 50: provide search/replace blocks in `text_or_search_replace_blocks`.
- Use absolute paths only (~ allowed).
"""
        + diffinstructions,
        annotations=ToolAnnotations(
            destructiveHint=True, idempotentHint=True, openWorldHint=False
        ),
    ),
    Tool(
        inputSchema=remove_titles_from_schema(ContextSave.model_json_schema()),
        name="ContextSave",
        description="""Save task context and relevant files for later resumption.
- Set `id` to a unique identifier (3 random words or user-provided).
- Set `project_root_path` to the project root, or empty string if unknown.
- Set `description` with detailed task context in markdown.
- Set `relevant_file_globs` to file paths or glob patterns to include.
""",
        annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False),
    ),
]
