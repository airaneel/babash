"""File writing and editing operations."""

import os
from hashlib import sha256
from pathlib import Path
from typing import Optional

from pydantic import ValidationError
from syntax_checker import Output as SCOutput
from syntax_checker import check_syntax as raw_check_syntax
from wcmatch import glob as wcglob

from ...types_ import (
    FileEdit,
    FileWriteOrEdit,
    ReadFiles,
    WriteIfEmpty,
)
from ..file_ops.extensions import select_max_tokens
from ..file_ops.search_replace import SEARCH_MARKER, search_replace_edit
from ..repo_ops.file_stats import FileStats, load_workspace_stats, save_workspace_stats
from .context import Context, default_enc, expand_user
from .read_ops import read_file, read_files


def check_syntax(ext: str, content: str) -> SCOutput:
    if ext == "html":
        return raw_check_syntax("html", "")
    return raw_check_syntax(ext, content)


def get_context_for_errors(
    errors: list[tuple[int, int]],
    file_content: str,
    filename: str,
    coding_max_tokens: Optional[int],
    noncoding_max_tokens: Optional[int],
) -> str:
    file_lines = file_content.split("\n")
    min_line_num = max(0, min([error[0] for error in errors]) - 10)
    max_line_num = min(len(file_lines), max([error[0] for error in errors]) + 10)
    context_lines = file_lines[min_line_num:max_line_num]
    context = "\n".join(context_lines)

    max_tokens = select_max_tokens(filename, coding_max_tokens, noncoding_max_tokens)
    if max_tokens is not None and max_tokens > 0:
        ntokens = len(default_enc.encoder(context))
        if ntokens > max_tokens:
            return "Please re-read the file to understand the context"
    return f"Here's relevant snippet from the file where the syntax errors occured:\n<snippet>\n{context}\n</snippet>"


def write_file(
    writefile: WriteIfEmpty,
    error_on_exist: bool,
    coding_max_tokens: Optional[int],
    noncoding_max_tokens: Optional[int],
    context: Context,
) -> tuple[str, dict[str, list[tuple[int, int]]]]:
    path_ = expand_user(writefile.file_path)

    workspace_path = context.bash_state.workspace_root
    stats = load_workspace_stats(workspace_path)
    if path_ not in stats.files:
        stats.files[path_] = FileStats()
    stats.files[path_].increment_write()
    save_workspace_stats(workspace_path, stats)

    if not os.path.isabs(path_):
        return (
            f"Failure: file_path should be absolute path, current working directory is {context.bash_state.cwd}",
            {},
        )

    error_on_exist_ = (
        error_on_exist and path_ not in context.bash_state.whitelist_for_overwrite
    )
    curr_hash = ""
    if error_on_exist and path_ in context.bash_state.whitelist_for_overwrite:
        if os.path.exists(path_):
            with open(path_, "rb") as f:
                file_content = f.read()
                curr_hash = sha256(file_content).hexdigest()
                whitelist_data = context.bash_state.whitelist_for_overwrite[path_]
                if curr_hash != whitelist_data.file_hash:
                    error_on_exist_ = True
                elif not whitelist_data.is_read_enough():
                    error_on_exist_ = True

    allowed_globs = context.bash_state.write_if_empty_mode.allowed_globs
    if allowed_globs != "all" and not wcglob.globmatch(
        path_, allowed_globs, flags=wcglob.GLOBSTAR
    ):
        return (
            f"Error: updating file {path_} not allowed in current mode. Doesn't match allowed globs: {allowed_globs}",
            {},
        )

    if (error_on_exist or error_on_exist_) and os.path.exists(path_):
        content = Path(path_).read_text().strip()
        if content and error_on_exist_:
            if path_ not in context.bash_state.whitelist_for_overwrite:
                msg = f"Error: you need to read existing file {path_} at least once before it can be overwritten.\n\n"
                file_content_str, truncated, _, _, line_range = read_file(
                    path_, coding_max_tokens, noncoding_max_tokens, context, False
                )
                final_message = "You can now safely retry writing immediately considering the above information." if not truncated else ""
                return (
                    msg + f"Here's the existing file:\n<file-contents-numbered>\n{file_content_str}\n{final_message}\n</file-contents-numbered>",
                    {path_: [line_range]},
                )

            whitelist_data = context.bash_state.whitelist_for_overwrite[path_]

            if curr_hash != whitelist_data.file_hash:
                msg = "Error: the file has changed since last read.\n\n"
                file_content_str, truncated, _, _, line_range = read_file(
                    path_, coding_max_tokens, noncoding_max_tokens, context, False
                )
                final_message = "You can now safely retry writing immediately considering the above information." if not truncated else ""
                return (
                    msg + f"Here's the existing file:\n<file-contents-numbered>\n{file_content_str}\n</file-contents-numbered>\n{final_message}",
                    {path_: [line_range]},
                )

            unread_ranges = whitelist_data.get_unread_ranges()
            ranges_str = ", ".join([f"{start}-{end}" for start, end in unread_ranges])
            msg = f"Error: you need to read more of the file before it can be overwritten.\nUnread line ranges: {ranges_str}\n\n"
            paths_: list[str] = [path_ + ":" + f"{start}-{end}" for start, end in unread_ranges]
            paths_readfiles = ReadFiles(file_paths=paths_)
            readfiles, file_ranges_dict, truncated = read_files(
                paths_readfiles.file_paths, coding_max_tokens, noncoding_max_tokens, context,
                start_line_nums=paths_readfiles.start_line_nums, end_line_nums=paths_readfiles.end_line_nums,
            )
            final_message = "Now that you have read the rest of the file, you can now safely immediately retry writing but consider the new information above." if not truncated else ""
            return (msg + "\n" + readfiles + "\n" + final_message), file_ranges_dict

    path = Path(path_)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("w") as f:
            f.write(writefile.file_content)
    except OSError as e:
        return f"Error: {e}", {}

    extension = Path(path_).suffix.lstrip(".")
    context.console.print(f"File written to {path_}")

    warnings = []
    try:
        check = check_syntax(extension, writefile.file_content)
        syntax_errors = check.description
        if syntax_errors:
            if extension in {"tsx", "ts"}:
                syntax_errors += "\nNote: Ignore if 'tagged template literals' are used, they may raise false positive errors in tree-sitter."
            context_for_errors = get_context_for_errors(
                check.errors, writefile.file_content, path_, coding_max_tokens, noncoding_max_tokens,
            )
            context.console.print(f"W: Syntax errors encountered: {syntax_errors}")
            warnings.append(f"""
---
Warning: tree-sitter reported syntax errors
Syntax errors:
{syntax_errors}

{context_for_errors}
---
            """)
    except Exception:
        pass

    total_lines = writefile.file_content.count("\n") + 1
    return "Success" + "".join(warnings), {path_: [(1, total_lines)]}


def do_diff_edit(
    fedit: FileEdit,
    coding_max_tokens: Optional[int],
    noncoding_max_tokens: Optional[int],
    context: Context,
) -> tuple[str, dict[str, list[tuple[int, int]]]]:
    try:
        return _do_diff_edit(fedit, coding_max_tokens, noncoding_max_tokens, context)
    except Exception as e:
        try:
            fedit = FileEdit(
                file_path=fedit.file_path,
                file_edit_using_search_replace_blocks=fedit.file_edit_using_search_replace_blocks.replace('\\"', '"'),
            )
            return _do_diff_edit(fedit, coding_max_tokens, noncoding_max_tokens, context)
        except Exception:
            pass
        raise e


def _do_diff_edit(
    fedit: FileEdit,
    coding_max_tokens: Optional[int],
    noncoding_max_tokens: Optional[int],
    context: Context,
) -> tuple[str, dict[str, list[tuple[int, int]]]]:
    context.console.log(f"Editing file: {fedit.file_path}")
    path_ = expand_user(fedit.file_path)

    if not os.path.isabs(path_):
        raise Exception(
            f"Failure: file_path should be absolute path, current working directory is {context.bash_state.cwd}"
        )

    workspace_path = context.bash_state.workspace_root
    stats = load_workspace_stats(workspace_path)
    if path_ not in stats.files:
        stats.files[path_] = FileStats()
    stats.files[path_].increment_edit()
    save_workspace_stats(workspace_path, stats)

    allowed_globs = context.bash_state.file_edit_mode.allowed_globs
    if allowed_globs != "all" and not wcglob.globmatch(
        path_, allowed_globs, flags=wcglob.GLOBSTAR
    ):
        raise Exception(
            f"Error: updating file {path_} not allowed in current mode. Doesn't match allowed globs: {allowed_globs}"
        )

    if not os.path.exists(path_):
        raise Exception(f"Error: file {path_} does not exist")

    with open(path_) as f:
        apply_diff_to = f.read()

    fedit.file_edit_using_search_replace_blocks = fedit.file_edit_using_search_replace_blocks.strip()
    edit_lines = fedit.file_edit_using_search_replace_blocks.split("\n")

    apply_diff_to, comments = search_replace_edit(edit_lines, apply_diff_to, context.console.log)

    total_lines = apply_diff_to.count("\n") + 1

    with open(path_, "w") as f:
        f.write(apply_diff_to)

    extension = Path(path_).suffix.lstrip(".")
    try:
        check = check_syntax(extension, apply_diff_to)
        syntax_errors = check.description
        if syntax_errors:
            context_for_errors = get_context_for_errors(
                check.errors, apply_diff_to, path_, coding_max_tokens, noncoding_max_tokens,
            )
            if extension in {"tsx", "ts"}:
                syntax_errors += "\nNote: Ignore if 'tagged template literals' are used, they may raise false positive errors in tree-sitter."
            context.console.print(f"W: Syntax errors encountered: {syntax_errors}")
            return (
                f"""{comments}
---
Warning: tree-sitter reported syntax errors, please re-read the file and fix if there are any errors.
Syntax errors:
{syntax_errors}

{context_for_errors}
""",
                {path_: [(1, total_lines)]},
            )
    except Exception:
        pass

    return comments, {path_: [(1, total_lines)]}


def _is_edit(content: str, percentage: int) -> bool:
    lines = content.lstrip().split("\n")
    if not lines:
        return False
    line = lines[0]
    if SEARCH_MARKER.match(line) or (0 < percentage <= 50):
        return True
    return False


def file_writing(
    file_writing_args: FileWriteOrEdit,
    coding_max_tokens: Optional[int],
    noncoding_max_tokens: Optional[int],
    context: Context,
) -> tuple[str, dict[str, list[tuple[int, int]]]]:
    if file_writing_args.thread_id and file_writing_args.thread_id != context.bash_state.current_thread_id:
        if not context.bash_state.load_state_from_thread_id(file_writing_args.thread_id):
            return (
                f"Error: No saved bash state found for thread_id `{file_writing_args.thread_id}`. Please re-initialize to get a new id or use correct id.",
                {},
            )

    path_ = expand_user(file_writing_args.file_path)
    if not os.path.isabs(path_):
        return (
            f"Failure: file_path should be absolute path, current working directory is {context.bash_state.cwd}",
            {},
        )

    content = file_writing_args.text_or_search_replace_blocks

    if not _is_edit(content, file_writing_args.percentage_to_change):
        return write_file(
            WriteIfEmpty(file_path=path_, file_content=content),
            True, coding_max_tokens, noncoding_max_tokens, context,
        )

    return do_diff_edit(
        FileEdit(file_path=path_, file_edit_using_search_replace_blocks=content),
        coding_max_tokens, noncoding_max_tokens, context,
    )
