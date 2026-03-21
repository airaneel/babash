import os

from mcp.types import Tool, ToolAnnotations

from ..types_ import (
    ContextSave,
    FileWriteOrEdit,
    Initialize,
    ReadFiles,
    ReadImage,
)

with open(os.path.join(os.path.dirname(__file__), "diff-instructions.txt")) as f:
    diffinstructions = f.read()


TOOL_PROMPTS = [
    Tool(
        inputSchema=Initialize.model_json_schema(),
        name="Initialize",
        description="""Initialize the shell environment. Must be called first before any other tool.
- Set `type` to "first_call" for initial setup, "user_asked_mode_change" to switch modes, "reset_shell" if shell is broken, "user_asked_change_workspace" to change directory.
- Optionally set `any_workspace_path` to the project directory.
- Optionally set `initial_files_to_read` to files the user mentioned.
- Optionally set `task_id_to_resume` to resume a previous task.
- Optionally set `mode_name` to "architect" (read-only) or "code_writer" (restricted). Default is full access.
""",
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    ),
    Tool(
        inputSchema={
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."},
                "is_background": {"type": "boolean", "default": False, "description": "Run in background shell."},
                "wait_for_seconds": {"type": "number", "description": "Max seconds to wait for output."},
            },
        },
        name="RunCommand",
        description="""Execute a shell command.
- Only one foreground command runs at a time. Check status with CheckStatus before running a new one.
- Set `is_background` to true for long-running commands. Returns a bg_command_id.
- Do not use echo/cat to read/write files — use ReadFiles/FileWriteOrEdit instead.
""",
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True),
    ),
    Tool(
        inputSchema={
            "type": "object",
            "properties": {
                "bg_command_id": {"type": "string", "description": "Background command ID. Omit to check the main shell."},
            },
        },
        name="CheckStatus",
        description="Check if a command is still running. Returns current output and status.",
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    ),
    Tool(
        inputSchema={
            "type": "object",
            "required": ["text"],
            "properties": {
                "text": {"type": "string", "description": "Text to send to stdin of the running program."},
                "bg_command_id": {"type": "string", "description": "Background command ID. Omit for main shell."},
            },
        },
        name="SendInput",
        description="Send text input to a running interactive program (e.g. password prompt, interactive CLI).",
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False),
    ),
    Tool(
        inputSchema={
            "type": "object",
            "required": ["keys"],
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["Enter", "Key-up", "Key-down", "Key-left", "Key-right", "Ctrl-c", "Ctrl-d"]},
                    "description": "Special keys to send.",
                },
                "bg_command_id": {"type": "string", "description": "Background command ID. Omit for main shell."},
            },
        },
        name="SendKeys",
        description="Send special keys to a running program. Use Ctrl-c to interrupt, arrow keys to navigate, Enter to confirm.",
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False),
    ),
    Tool(
        inputSchema=ReadFiles.model_json_schema(),
        name="ReadFiles",
        description="""Read content of one or more files.
- Provide absolute paths only (~ allowed).
- Supports line ranges: `/path/file.py:10-20` for lines 10-20, `:10-` from line 10, `:-20` first 20 lines.
""",
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    ),
    Tool(
        inputSchema=ReadImage.model_json_schema(),
        name="ReadImage",
        description="Read an image file and return its contents. Provide absolute path.",
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    ),
    Tool(
        inputSchema=FileWriteOrEdit.model_json_schema(),
        name="FileWriteOrEdit",
        description="""Write or edit a file.
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
        inputSchema=ContextSave.model_json_schema(),
        name="ContextSave",
        description="""Save task context and relevant files for later resumption.
- Set `id` to a unique identifier (3 random words or user-provided).
- Set `project_root_path` to the project root, or empty string if unknown.
- Set `description` with detailed task context in markdown.
- Set `relevant_file_globs` to file paths or glob patterns to include.
""",
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    ),
]
