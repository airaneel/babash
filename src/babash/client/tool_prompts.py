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
- Set `type` to "command" and provide `command` to run a shell command.
- Set `type` to "status_check" to check if a previous command is still running.
- Set `type` to "send_text" to send text input to a running interactive program.
- Set `type` to "send_specials" to send special keys like Enter, Ctrl-c, arrow keys.
- Set `type` to "send_ascii" to send raw ASCII codes.
- Set `thread_id` to the value returned by Initialize.
- Set `is_background` to true to run a command in background (gets its own bg_command_id).
- Set `bg_command_id` when interacting with a background command.
- Only one foreground command runs at a time. Check status before running a new one.
- Do not use echo/cat to read/write files — use ReadFiles/FileWriteOrEdit instead.
- Do not send Ctrl-c without checking status first. Programs may still be running.
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
