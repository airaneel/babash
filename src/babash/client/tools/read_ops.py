"""File reading operations."""

import os
from pathlib import Path
from typing import Optional

from ...types_ import ReadFiles
from ..bash_state import BashState
from ..file_ops.extensions import select_max_tokens
from ..repo_ops.file_stats import FileStats, load_workspace_stats, save_workspace_stats
from .context import Context, default_enc, expand_user, read_image_from_shell, ImageData


def range_format(start_line_num: Optional[int], end_line_num: Optional[int]) -> str:
    st = "" if not start_line_num else str(start_line_num)
    end = "" if not end_line_num else str(end_line_num)
    if not st and not end:
        return ""
    return f":{st}-{end}"


def read_files(
    file_paths: list[str],
    coding_max_tokens: Optional[int],
    noncoding_max_tokens: Optional[int],
    context: Context,
    start_line_nums: Optional[list[Optional[int]]] = None,
    end_line_nums: Optional[list[Optional[int]]] = None,
) -> tuple[str, dict[str, list[tuple[int, int]]], bool]:
    message = ""
    file_ranges_dict: dict[str, list[tuple[int, int]]] = {}

    workspace_path = context.bash_state.workspace_root
    stats = load_workspace_stats(workspace_path)

    for path_ in file_paths:
        path_ = expand_user(path_)
        if not os.path.isabs(path_):
            continue
        if path_ not in stats.files:
            stats.files[path_] = FileStats()
        stats.files[path_].increment_read()
    save_workspace_stats(workspace_path, stats)

    truncated = False
    for i, file in enumerate(file_paths):
        try:
            start_line_num = None if start_line_nums is None else start_line_nums[i]
            end_line_num = None if end_line_nums is None else end_line_nums[i]

            content, truncated, tokens, path, line_range = read_file(
                file, coding_max_tokens, noncoding_max_tokens, context,
                start_line_num, end_line_num,
            )

            if path in file_ranges_dict:
                file_ranges_dict[path].append(line_range)
            else:
                file_ranges_dict[path] = [line_range]
        except Exception as e:
            message += f"\n{file}: {str(e)}\n"
            continue

        if coding_max_tokens:
            coding_max_tokens = max(0, coding_max_tokens - tokens)
        if noncoding_max_tokens:
            noncoding_max_tokens = max(0, noncoding_max_tokens - tokens)

        range_formatted = range_format(start_line_num, end_line_num)
        message += (
            f'\n<file-contents-numbered path="{file}{range_formatted}">\n{content}\n'
        )

        if not truncated:
            message += "</file-contents-numbered>"

        if (
            truncated
            or (coding_max_tokens is not None and coding_max_tokens <= 0)
            and (noncoding_max_tokens is not None and noncoding_max_tokens <= 0)
        ):
            not_reading = file_paths[i + 1 :]
            if not_reading:
                message += f"\nNot reading the rest of the files: {', '.join(not_reading)} due to token limit, please call again"
            break

    return message, file_ranges_dict, truncated


def read_file(
    file_path: str,
    coding_max_tokens: Optional[int],
    noncoding_max_tokens: Optional[int],
    context: Context,
    start_line_num: Optional[int] = None,
    end_line_num: Optional[int] = None,
) -> tuple[str, bool, int, str, tuple[int, int]]:
    context.console.print(f"Reading file: {file_path}")

    file_path = expand_user(file_path)

    if not os.path.isabs(file_path):
        raise ValueError(
            f"Failure: file_path should be absolute path, current working directory is {context.bash_state.cwd}"
        )

    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"Error: file {file_path} does not exist")

    with path.open("r") as f:
        all_lines = f.readlines(10_000_000)
        if all_lines and all_lines[-1].endswith("\n"):
            all_lines.append("")

    total_lines = len(all_lines)

    start_idx = 0
    if start_line_num is not None:
        start_idx = max(0, start_line_num - 1)

    end_idx = len(all_lines)
    if end_line_num is not None:
        end_idx = min(len(all_lines), end_line_num)

    effective_start = start_line_num if start_line_num is not None else 1
    effective_end = end_line_num if end_line_num is not None else total_lines

    filtered_lines = all_lines[start_idx:end_idx]

    content_lines = []
    for i, line in enumerate(filtered_lines, start=start_idx + 1):
        content_lines.append(f"{i} {line}")
    content = "".join(content_lines)

    truncated = False
    tokens_counts = 0

    max_tokens = select_max_tokens(file_path, coding_max_tokens, noncoding_max_tokens)

    if max_tokens is not None:
        tokens = default_enc.encoder(content)
        tokens_counts = len(tokens)

        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            truncated_content = default_enc.decoder(truncated_tokens)
            line_count = truncated_content.count("\n")
            last_line_shown = start_idx + line_count

            content = truncated_content
            total_lines = len(all_lines)
            content += (
                f"\n(...truncated) Only showing till line number {last_line_shown} of {total_lines} total lines due to the token limit, please continue reading from {last_line_shown + 1} if required"
                f" using syntax {file_path}:{last_line_shown + 1}-{total_lines}"
            )
            truncated = True
            effective_end = last_line_shown

    return (content, truncated, tokens_counts, file_path, (effective_start, effective_end))
