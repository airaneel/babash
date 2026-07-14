"""Model-based tool dispatch — TEST SCAFFOLDING ONLY.

The MCP server uses native @mcp.tool() handlers with typed parameters and never
calls this. It used to ship inside the package, where it was 163 lines of code
no production path executed. It lives here now so the tests can keep driving the
underlying ops (execute_bash, file_writing, read_files, …) through one entry
point, without the package carrying a second, unused API surface."""

from typing import Any, Callable, Optional, Type

from pydantic import TypeAdapter, ValidationError

from babash.types_ import (
    BashCommand,
    ContextSave,
    FileEdit,
    FileWriteOrEdit,
    Initialize,
    ReadFiles,
    ReadImage,
    WriteIfEmpty,
)
from babash.client.bash_state import execute_bash
from babash.client.encoder import EncoderDecoder
from babash.client.tools.context import Context, ImageData
from babash.client.tools.init_ops import _handle_context_save, is_mode_change
from babash.client.tools.read_ops import read_files
from babash.client.tools.write_ops import do_diff_edit, file_writing, write_file

import json
import os

TOOLS = BashCommand | FileWriteOrEdit | ReadImage | ReadFiles | Initialize | ContextSave


def which_tool_name(name: str) -> Type[TOOLS]:
    registry: dict[str, Type[TOOLS]] = {
        "BashCommand": BashCommand,
        "FileWriteOrEdit": FileWriteOrEdit,
        "ReadImage": ReadImage,
        "ReadFiles": ReadFiles,
        "Initialize": Initialize,
        "ContextSave": ContextSave,
    }
    if name not in registry:
        raise ValueError(f"Unknown tool name: {name}")
    return registry[name]


def parse_tool_by_name(name: str, arguments: dict[str, Any]) -> TOOLS:
    tool_type = which_tool_name(name)
    try:
        return tool_type(**arguments)
    except ValidationError:
        def try_json(x: str) -> Any:
            if not isinstance(x, str):
                return x
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                return x
        return tool_type(**{k: try_json(v) for k, v in arguments.items()})


def _merge_ranges(
    target: dict[str, list[tuple[int, int]]],
    source: dict[str, list[tuple[int, int]]],
) -> None:
    for path, ranges in source.items():
        if path in target:
            target[path].extend(ranges)
        else:
            target[path] = list(ranges)


def get_tool_output(
    context: Context,
    args: dict[object, object] | TOOLS,
    enc: EncoderDecoder[int],
    limit: float,
    loop_call: Callable[[str, float], tuple[str, float]],
    coding_max_tokens: Optional[int],
    noncoding_max_tokens: Optional[int],
) -> tuple[list[str | ImageData], float]:
    if isinstance(args, dict):
        adapter = TypeAdapter[TOOLS](TOOLS, config={"extra": "forbid"})
        arg = adapter.validate_python(args)
    else:
        arg = args

    output: tuple[str | ImageData, float]
    file_paths_with_ranges: dict[str, list[tuple[int, int]]] = {}

    if isinstance(arg, BashCommand):
        output_str, cost = execute_bash(
            context.bash_state, enc, arg,
            noncoding_max_tokens, arg.action_json.wait_for_seconds,
        )
        output = output_str, cost

    elif isinstance(arg, WriteIfEmpty):
        result, paths = write_file(arg, True, coding_max_tokens, noncoding_max_tokens, context)
        output = result, 0.0
        _merge_ranges(file_paths_with_ranges, paths)

    elif isinstance(arg, FileEdit):
        result, paths = do_diff_edit(arg, coding_max_tokens, noncoding_max_tokens, context)
        output = result, 0.0
        _merge_ranges(file_paths_with_ranges, paths)

    elif isinstance(arg, FileWriteOrEdit):
        result, paths = file_writing(arg, coding_max_tokens, noncoding_max_tokens, context)
        output = result, 0.0
        _merge_ranges(file_paths_with_ranges, paths)

    elif isinstance(arg, ReadImage):
        from babash.client.tools.context import read_image_from_shell
        image_data = read_image_from_shell(arg.file_path, context)
        output = image_data, 0.0

    elif isinstance(arg, ReadFiles):
        result, paths, _ = read_files(
            arg.file_paths, coding_max_tokens, noncoding_max_tokens, context,
            arg.start_line_nums, arg.end_line_nums,
        )
        output = result, 0.0
        _merge_ranges(file_paths_with_ranges, paths)

    elif isinstance(arg, Initialize):
        if arg.type in ("user_asked_mode_change", "reset_shell"):
            workspace_path = (
                arg.any_workspace_path
                if os.path.isdir(arg.any_workspace_path)
                else os.path.dirname(arg.any_workspace_path)
            )
            workspace_path = workspace_path if os.path.exists(workspace_path) else ""
            from babash.client.tools.init_ops import reset_babash
            result = reset_babash(
                context, workspace_path,
                arg.mode_name if is_mode_change(arg.mode, context.bash_state) else None,
                arg.mode, arg.thread_id,
            )
            output = result, 0.0
        else:
            from babash.client.tools.init_ops import initialize
            from typing import Literal
            init_type: Literal["user_asked_change_workspace", "first_call"] = arg.type  # type: ignore[assignment]
            output_, context, init_paths = initialize(
                init_type, context, arg.any_workspace_path,
                arg.initial_files_to_read or [], arg.task_id_to_resume,
                coding_max_tokens, noncoding_max_tokens, arg.mode, arg.thread_id,
            )
            output = output_, 0.0
            _merge_ranges(file_paths_with_ranges, {
                p: r for p, r in init_paths.items() if os.path.exists(p)
            })

    elif isinstance(arg, ContextSave):
        result = _handle_context_save(arg, context)
        output = result, 0.0

    else:
        raise ValueError(f"Unknown tool: {arg}")

    if file_paths_with_ranges:
        context.bash_state.add_to_whitelist_for_overwrite(file_paths_with_ranges)
    context.bash_state.save_state_to_disk()

    return [output[0]], output[1]
