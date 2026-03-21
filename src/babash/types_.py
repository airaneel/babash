import os
import re
from typing import Annotated, Any, Literal, Optional, Protocol, Sequence, Union

from pydantic import BaseModel as PydanticBaseModel
from pydantic import (
    Discriminator,
    Field,
    PrivateAttr,
    Tag,
    model_serializer,
    model_validator,
)


def normalize_thread_id(thread_id: str) -> str:
    """Normalize thread_id by keeping only word characters (alphanumeric and underscore)."""
    return re.sub(r"[^\w]", "", thread_id)


def _patch_singleton_all(
    value: Literal["all"] | list[str],
) -> Literal["all"] | list[str]:
    """Patch ["all"] to "all" — handles frequent LLM output quirk."""
    if isinstance(value, list) and len(value) == 1 and value[0] == "all":
        return "all"
    return value


class NoExtraArgs(PydanticBaseModel):
    class Config:
        extra = "forbid"


BaseModel = NoExtraArgs


Modes = Literal["babash", "architect", "code_writer"]


class CodeWriterMode(BaseModel):
    allowed_globs: Literal["all"] | list[str]
    allowed_commands: Literal["all"] | list[str]

    def model_post_init(self, _: Any) -> None:
        self.allowed_commands = _patch_singleton_all(self.allowed_commands)
        self.allowed_globs = _patch_singleton_all(self.allowed_globs)

    def update_relative_globs(self, workspace_root: str) -> None:
        """Update globs if they're relative paths"""
        if self.allowed_globs != "all":
            self.allowed_globs = [
                glob if os.path.isabs(glob) else os.path.join(workspace_root, glob)
                for glob in self.allowed_globs
            ]


ModesConfig = Union[Literal["babash", "architect"], CodeWriterMode]


class Initialize(BaseModel):
    type: Literal[
        "first_call",
        "user_asked_mode_change",
        "reset_shell",
        "user_asked_change_workspace",
    ]
    any_workspace_path: str = Field(
        default="",
        description="Project directory to initialize in. Optional.",
    )
    initial_files_to_read: list[str] = Field(
        default_factory=list,
        description="Files to read on init. Optional.",
    )
    task_id_to_resume: str = Field(
        default="",
        description="Task ID to resume from a previous session. Leave empty for new tasks.",
    )
    mode_name: Literal["babash", "architect", "code_writer"] = Field(
        default="babash",
        description="Execution mode.",
    )
    thread_id: str = Field(
        default="",
        description="Thread ID from a previous Initialize call. Leave empty on first_call.",
    )
    allowed_globs: Optional[Literal["all"] | list[str]] = Field(
        default=None,
        description="File globs that are allowed to be edited. Set to 'all' to allow all files, or provide a list of glob patterns. Only required when mode_name is 'code_writer'.",
    )
    allowed_commands: Optional[Literal["all"] | list[str]] = Field(
        default=None,
        description="Shell commands that are allowed to be executed. Set to 'all' to allow all commands, or provide a list of command patterns. Only required when mode_name is 'code_writer'.",
    )

    def model_post_init(self, __context: Any) -> None:
        self.thread_id = normalize_thread_id(self.thread_id)
        if self.mode_name == "code_writer":
            assert self.allowed_globs is not None, (
                "allowed_globs can't be null when the mode is code_writer"
            )
            assert self.allowed_commands is not None, (
                "allowed_commands can't be null when the mode is code_writer"
            )
            self.allowed_commands = _patch_singleton_all(self.allowed_commands)
            self.allowed_globs = _patch_singleton_all(self.allowed_globs)
        if self.type != "first_call" and not self.thread_id:
            raise ValueError(
                "Thread id should be provided if type != 'first_call', including when resetting"
            )
        return super().model_post_init(__context)

    @property
    def mode(self) -> ModesConfig:
        if self.mode_name != "code_writer":
            return self.mode_name
        assert self.allowed_globs is not None
        assert self.allowed_commands is not None
        return CodeWriterMode(
            allowed_globs=self.allowed_globs, allowed_commands=self.allowed_commands
        )

    def update_relative_globs(self, workspace_root: str) -> None:
        """Update globs if they're relative paths"""
        if self.allowed_globs is not None and self.allowed_globs != "all":
            self.allowed_globs = [
                glob if os.path.isabs(glob) else os.path.join(workspace_root, glob)
                for glob in self.allowed_globs
            ]


class CommandBase(PydanticBaseModel):
    """Base for bash action types. Allows extra fields so LLM mistakes
    (e.g. sending 'command' with type='status_check') are silently ignored
    rather than causing validation errors."""

    class Config:
        extra = "ignore"

    wait_for_seconds: Optional[float] = None
    thread_id: str = ""

    def model_post_init(self, __context: Any) -> None:
        if self.thread_id:
            self.thread_id = normalize_thread_id(self.thread_id)
        return super().model_post_init(__context)


class Command(CommandBase):
    command: str
    type: Literal["command"] = "command"
    is_background: bool = False


class StatusCheck(CommandBase):
    status_check: Literal[True] = True
    type: Literal["status_check"] = "status_check"
    bg_command_id: str | None = None


class SendText(CommandBase):
    send_text: str
    type: Literal["send_text"] = "send_text"
    bg_command_id: str | None = None


Specials = Literal[
    "Enter", "Key-up", "Key-down", "Key-left", "Key-right", "Ctrl-c", "Ctrl-d"
]


class SendSpecials(CommandBase):
    send_specials: Sequence[Specials]
    type: Literal["send_specials"] = "send_specials"
    bg_command_id: str | None = None


class SendAscii(CommandBase):
    send_ascii: Sequence[int]
    type: Literal["send_ascii"] = "send_ascii"
    bg_command_id: str | None = None


_BASH_COMMAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["type"],
    "properties": {
        "type": {
            "type": "string",
            "enum": ["command", "status_check", "send_text", "send_specials", "send_ascii"],
            "description": "Action type. Determines which field to set.",
        },
        "wait_for_seconds": {"type": "number", "description": "Optional timeout."},
    },
    "oneOf": [
        {
            "title": "Run a shell command",
            "properties": {
                "type": {"const": "command"},
                "command": {"type": "string", "description": "Shell command to execute."},
                "is_background": {"type": "boolean", "default": False, "description": "Run in background."},
            },
            "required": ["command"],
        },
        {
            "title": "Check status of a running command",
            "properties": {
                "type": {"const": "status_check"},
                "bg_command_id": {"type": "string", "description": "Background command ID to check."},
            },
        },
        {
            "title": "Send text input to a running program",
            "properties": {
                "type": {"const": "send_text"},
                "send_text": {"type": "string", "description": "Text to send to stdin."},
                "bg_command_id": {"type": "string", "description": "Background command ID."},
            },
            "required": ["send_text"],
        },
        {
            "title": "Send special keys",
            "properties": {
                "type": {"const": "send_specials"},
                "send_specials": {
                    "oneOf": [
                        {"type": "array", "items": {"type": "string", "enum": ["Enter", "Key-up", "Key-down", "Key-left", "Key-right", "Ctrl-c", "Ctrl-d"]}},
                        {"type": "string"},
                    ],
                    "description": "Special keys to send. Array like [\"Ctrl-c\"] or single string.",
                },
                "bg_command_id": {"type": "string", "description": "Background command ID."},
            },
            "required": ["send_specials"],
        },
        {
            "title": "Send raw ASCII codes",
            "properties": {
                "type": {"const": "send_ascii"},
                "send_ascii": {
                    "oneOf": [
                        {"type": "array", "items": {"type": "integer"}},
                        {"type": "string"},
                    ],
                    "description": "ASCII codes to send. Array like [3] or string.",
                },
                "bg_command_id": {"type": "string", "description": "Background command ID."},
            },
            "required": ["send_ascii"],
        },
    ],
}


def _bash_action_discriminator(data: Any) -> str:
    if isinstance(data, dict):
        return str(data.get("type", "command"))
    return str(getattr(data, "type", "command"))


BashAction = Annotated[
    Annotated[Command, Tag("command")]
    | Annotated[StatusCheck, Tag("status_check")]
    | Annotated[SendText, Tag("send_text")]
    | Annotated[SendSpecials, Tag("send_specials")]
    | Annotated[SendAscii, Tag("send_ascii")],
    Discriminator(_bash_action_discriminator),
]


def _fix_llm_bash_mistakes(data: dict[str, Any]) -> dict[str, Any]:
    """Fix common LLM mistakes in BashCommand arguments."""
    import json as _json

    action_type = data.get("type", "command")

    # Fix stringified arrays: "[3]" -> [3], "[\"Ctrl-c\"]" -> ["Ctrl-c"]
    # Also wrap bare values: "Ctrl-c" -> ["Ctrl-c"], "3" -> [3]
    for field in ("send_specials", "send_ascii"):
        val = data.get(field)
        if not isinstance(val, str):
            continue
        try:
            parsed = _json.loads(val)
            if isinstance(parsed, list):
                data = {**data, field: parsed}
            elif isinstance(parsed, (int, float)):
                data = {**data, field: [int(parsed)]}
            else:
                data = {**data, field: [str(parsed)]}
        except _json.JSONDecodeError:
            data = {**data, field: [val]}

    # Fix 'command' field used for non-command types
    cmd = data.get("command")
    if cmd and action_type != "command":
        field_map = {
            "send_text": "send_text",
            "send_specials": "send_specials",
            "send_ascii": "send_ascii",
        }
        target = field_map.get(action_type)
        if target and target not in data:
            data = {**data, target: cmd}

    return data


class BashCommand(BaseModel):
    action_json: BashAction

    @model_validator(mode="before")
    @classmethod
    def combine(cls, data: Any) -> Any:
        if isinstance(data, dict) and "action_json" in data:
            return data
        if isinstance(data, dict):
            data = _fix_llm_bash_mistakes(data)
        return {"action_json": data}

    @model_serializer(mode="plain")
    def serialize_model(self) -> dict[str, Any]:
        return self.action_json.model_dump()

    @staticmethod
    def model_json_schema(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return _BASH_COMMAND_SCHEMA


class ReadImage(BaseModel):
    file_path: str


class WriteIfEmpty(BaseModel):
    file_path: str
    file_content: str


class ReadFiles(BaseModel):
    file_paths: list[str]
    _start_line_nums: list[int | None] = PrivateAttr(default_factory=lambda: [])
    _end_line_nums: list[int | None] = PrivateAttr(default_factory=lambda: [])

    @property
    def show_line_numbers_reason(self) -> str:
        return "True"

    @property
    def start_line_nums(self) -> list[int | None]:
        """Get the start line numbers."""
        return self._start_line_nums

    @property
    def end_line_nums(self) -> list[int | None]:
        """Get the end line numbers."""
        return self._end_line_nums

    @staticmethod
    def _parse_line_range(file_path: str) -> tuple[str, int | None, int | None]:
        """Parse 'file.py:10-20' into (path, start, end). Returns original path on no match."""
        if ":" not in file_path:
            return file_path, None, None

        parts = file_path.rsplit(":", 1)
        if len(parts) != 2:
            return file_path, None, None

        path, spec = parts

        # file.py:10
        if spec.isdigit():
            return path, int(spec), None

        if "-" not in spec:
            return file_path, None, None

        left, right = spec.split("-", 1)

        # file.py:-20
        if not left and right.isdigit():
            return path, None, int(right)

        # file.py:10- or file.py:10-20
        if left.isdigit():
            end = int(right) if right.isdigit() else None
            return path, int(left), end

        return file_path, None, None

    def model_post_init(self, __context: Any) -> None:
        self._start_line_nums = []
        self._end_line_nums = []
        clean_file_paths = []

        for file_path in self.file_paths:
            path, start, end = self._parse_line_range(file_path)
            clean_file_paths.append(path)
            self._start_line_nums.append(start)
            self._end_line_nums.append(end)

        self.file_paths = clean_file_paths
        return super().model_post_init(__context)


class FileEdit(BaseModel):
    file_path: str
    file_edit_using_search_replace_blocks: str


class FileWriteOrEdit(BaseModel):
    # Naming should be in sorted order otherwise it gets changed in LLM backend.
    file_path: str = Field(description="#1: absolute file path")
    percentage_to_change: int = Field(
        description="#2: predict this percentage, calculated as number of existing lines that will have some diff divided by total existing lines."
    )
    text_or_search_replace_blocks: str = Field(
        description="#3: content/edit blocks. Must be after #2 in the tool xml"
    )
    thread_id: str = Field(default="", description="Auto-injected by server.")

    def model_post_init(self, __context: Any) -> None:
        if self.thread_id:
            self.thread_id = normalize_thread_id(self.thread_id)
        return super().model_post_init(__context)


class ContextSave(BaseModel):
    id: str
    project_root_path: str = Field(
        default="",
        description="Project root directory. Leave empty if unknown.",
    )
    description: str
    relevant_file_globs: list[str]


class Console(Protocol):
    def print(self, *objects: Any, **kwargs: Any) -> None: ...

    def log(self, *objects: Any, **kwargs: Any) -> None: ...


class Mdata(PydanticBaseModel):
    data: BashCommand | FileWriteOrEdit | str | ReadFiles | Initialize | ContextSave
