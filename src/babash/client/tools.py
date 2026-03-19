import base64
import glob
import json
import mimetypes
import os
import subprocess
import traceback
import uuid
from dataclasses import dataclass
from hashlib import sha256
from os.path import expanduser
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    ParamSpec,
    Type,
    TypeVar,
)

import rich
from pydantic import BaseModel, TypeAdapter, ValidationError
from syntax_checker import Output as SCOutput
from syntax_checker import check_syntax as raw_check_syntax
from wcmatch import glob as wcglob

from ..client.bash_state import (
    BashState,
    execute_bash,
    generate_thread_id,
    get_status,
    get_tmpdir,
)
from ..client.repo_ops.file_stats import (
    FileStats,
    load_workspace_stats,
    save_workspace_stats,
)
from ..types_ import (
    BashCommand,
    CodeWriterMode,
    Command,
    Console,
    ContextSave,
    FileEdit,
    FileWriteOrEdit,
    Initialize,
    Modes,
    ModesConfig,
    ReadFiles,
    ReadImage,
    WriteIfEmpty,
)
from .encoder import EncoderDecoder, get_default_encoder
from .file_ops.extensions import select_max_tokens
from .file_ops.search_replace import (
    SEARCH_MARKER,
    search_replace_edit,
)
from .memory import load_memory, save_memory
from .modes import (
    ARCHITECT_PROMPT,
    BABASH_PROMPT,
    code_writer_prompt,
    modes_to_state,
)
from .repo_ops.repo_context import get_repo_context


@dataclass
class Context:
    bash_state: BashState
    console: Console


def check_syntax(ext: str, content: str) -> SCOutput:
    if ext == "html":
        # Ignore due to prevelance of templating, causing issues
        return raw_check_syntax("html", "")
    return raw_check_syntax(ext, content)


def get_mode_prompt(context: Context) -> str:
    mode_prompt = ""
    if context.bash_state.mode == "code_writer":
        mode_prompt = code_writer_prompt(
            context.bash_state.file_edit_mode.allowed_globs,
            context.bash_state.write_if_empty_mode.allowed_globs,
            "all" if context.bash_state.bash_command_mode.allowed_commands else [],
        )
    elif context.bash_state.mode == "architect":
        mode_prompt = ARCHITECT_PROMPT
    else:
        mode_prompt = BABASH_PROMPT

    return mode_prompt


def _resume_task(
    task_id: str,
    is_first_call: bool,
    coding_max_tokens: Optional[int],
    noncoding_max_tokens: Optional[int],
) -> tuple[str, str, Optional[dict[str, Any]]]:
    """Returns (memory_text, workspace_path_override, loaded_state)."""
    if not task_id:
        return "", "", None
    if not is_first_call:
        return "Warning: task can only be resumed in a new conversation. No task loaded.", "", None
    try:
        project_root_path, task_mem, loaded_state = load_memory(
            task_id,
            coding_max_tokens,
            noncoding_max_tokens,
            lambda x: default_enc.encoder(x),
            lambda x: default_enc.decoder(x),
        )
        workspace = project_root_path if os.path.exists(project_root_path) else ""
        return "Following is the retrieved task:\n" + task_mem, workspace, loaded_state
    except Exception:
        return f'Error: Unable to load task with ID "{task_id}" ', "", None


def _resolve_workspace(
    workspace_path: str,
    is_first_call: bool,
    read_files_: list[str],
    mode: ModesConfig,
) -> tuple[str, Optional[Path], list[str]]:
    """Returns (repo_context, folder_to_start, read_files_)."""
    if is_first_call and not workspace_path:
        tmp_dir = get_tmpdir()
        workspace_path = os.path.join(tmp_dir, "claude-playground-" + uuid.uuid4().hex[:4])

    if not workspace_path:
        return "", None, read_files_

    if not os.path.exists(workspace_path):
        if os.path.abspath(workspace_path):
            os.makedirs(workspace_path, exist_ok=True)
            return f"\nInfo: Workspace path {workspace_path} did not exist. I've created it for you.\n", Path(workspace_path), read_files_
        return f"\nInfo: Workspace path {workspace_path} does not exist.", None, read_files_

    if os.path.isfile(workspace_path):
        if not read_files_:
            read_files_ = [workspace_path]
        workspace_path = os.path.dirname(workspace_path)

    repo_context, folder_to_start = get_repo_context(workspace_path)
    repo_context = f"---\n# Workspace structure\n{repo_context}\n---\n"

    if isinstance(mode, CodeWriterMode):
        mode.update_relative_globs(workspace_path)

    return repo_context, folder_to_start, read_files_


def _load_alignment_docs(folder_to_start: Optional[Path], console: Any) -> str:
    """Load CLAUDE.md/AGENTS.md from global and workspace dirs."""
    alignment = ""

    try:
        subprocess.run(["which", "rg"], timeout=1, capture_output=True, check=True)
        alignment += "---\n# Available commands\n\n- Use ripgrep `rg` command instead of `grep` because it's much much faster.\n\n---\n\n"
    except Exception:
        pass

    for base_dir, label in [
        (os.path.join(expanduser("~"), ".babash"), "Important guidelines from the user"),
        (str(folder_to_start) if folder_to_start else None, None),
    ]:
        if not base_dir:
            continue
        try:
            for fname in ("CLAUDE.md", "AGENTS.md"):
                fpath = os.path.join(base_dir, fname)
                if not os.path.exists(fpath):
                    continue
                with open(fpath, "r") as f:
                    content = f.read()
                heading = label or f"{fname} - user shared project guidelines to follow"
                alignment += f"---\n# {heading}\n```\n{content}\n```\n---\n\n"
                break
        except Exception as e:
            console.log(f"Error reading alignment file in {base_dir}: {e}")

    return alignment


def initialize(
    type: Literal["user_asked_change_workspace", "first_call"],
    context: Context,
    any_workspace_path: str,
    read_files_: list[str],
    task_id_to_resume: str,
    coding_max_tokens: Optional[int],
    noncoding_max_tokens: Optional[int],
    mode: ModesConfig,
    thread_id: str,
) -> tuple[str, Context, dict[str, list[tuple[int, int]]]]:
    any_workspace_path = expand_user(any_workspace_path)

    # Validate thread_id for non-first calls
    if type != "first_call" and thread_id != context.bash_state.current_thread_id:
        if not context.bash_state.load_state_from_thread_id(thread_id):
            return (
                f"Error: No saved bash state found for thread_id `{thread_id}`. Please re-initialize to get a new id or use correct id.",
                context,
                {},
            )

    # Resume task if requested
    memory, workspace_override, loaded_state = _resume_task(
        task_id_to_resume, type == "first_call", coding_max_tokens, noncoding_max_tokens
    )
    if workspace_override:
        any_workspace_path = workspace_override

    # Resolve workspace path
    repo_context, folder_to_start, read_files_ = _resolve_workspace(
        any_workspace_path, type == "first_call", read_files_, mode
    )

    # Apply mode to bash state
    if loaded_state is not None:
        try:
            snapshot = BashState.parse_state(loaded_state)
            workspace_root = str(folder_to_start) if folder_to_start else snapshot.workspace_root

            if mode == "babash":
                bcm, fem, wem, mn = snapshot.bash_command_mode, snapshot.file_edit_mode, snapshot.write_if_empty_mode, snapshot.mode
            else:
                mi = modes_to_state(mode)
                bcm, fem, wem, mn = mi.bash_command_mode, mi.file_edit_mode, mi.write_if_empty_mode, mi.mode_name

            cwd = str(folder_to_start) if folder_to_start else workspace_root
            whitelist = {**snapshot.whitelist_for_overwrite, **context.bash_state.whitelist_for_overwrite}
            context.bash_state.load_state(bcm, fem, wem, mn, whitelist, cwd, workspace_root,
                                          snapshot.thread_id or context.bash_state.current_thread_id)
        except ValueError:
            context.console.print(traceback.format_exc())
            context.console.print("Error: couldn't load bash state")
        mode_prompt = get_mode_prompt(context)
    else:
        mode_changed = is_mode_change(mode, context.bash_state)
        mode_impl = modes_to_state(mode)
        new_thread_id = generate_thread_id() if type == "first_call" else context.bash_state.current_thread_id
        folder_str = str(folder_to_start) if folder_to_start else ""
        context.bash_state.load_state(
            mode_impl.bash_command_mode, mode_impl.file_edit_mode, mode_impl.write_if_empty_mode,
            mode_impl.mode_name, dict(context.bash_state.whitelist_for_overwrite),
            folder_str, folder_str, new_thread_id,
        )
        mode_prompt = get_mode_prompt(context) if (type == "first_call" or mode_changed) else ""

    # Read initial files
    initial_files_context = ""
    initial_paths_with_ranges: dict[str, list[tuple[int, int]]] = {}
    if read_files_:
        if folder_to_start:
            read_files_ = [
                os.path.join(folder_to_start, f) if not os.path.isabs(expand_user(f)) else expand_user(f)
                for f in read_files_
            ]
        initial_files, initial_paths_with_ranges, _ = read_files(
            read_files_, coding_max_tokens, noncoding_max_tokens, context
        )
        initial_files_context = f"---\n# Requested files\nHere are the contents of the requested files:\n{initial_files}\n---\n"

    alignment_context = _load_alignment_docs(folder_to_start, context.console)

    output = f"""
Use thread_id={context.bash_state.current_thread_id} for all babash tool calls which take that.
---
{mode_prompt}

# Environment
System: {os.uname().sysname}
Machine: {os.uname().machine}
Initialized in directory (also cwd): {context.bash_state.cwd}
User home directory: {expanduser("~")}

{alignment_context}
{repo_context}

---

{memory}
---

{initial_files_context}

"""

    return output, context, initial_paths_with_ranges


def is_mode_change(mode_config: ModesConfig, bash_state: BashState) -> bool:
    mode_impl = modes_to_state(mode_config)
    return (
        mode_impl.bash_command_mode != bash_state.bash_command_mode
        or mode_impl.file_edit_mode != bash_state.file_edit_mode
        or mode_impl.write_if_empty_mode != bash_state.write_if_empty_mode
        or mode_impl.mode_name != bash_state.mode
    )


def reset_babash(
    context: Context,
    starting_directory: str,
    mode_name: Optional[Modes],
    change_mode: ModesConfig,
    thread_id: str,
) -> str:
    # Load state for this thread_id before proceeding with mode/directory changes
    if thread_id != context.bash_state.current_thread_id:
        # Try to load state from the thread_id
        if not context.bash_state.load_state_from_thread_id(thread_id):
            return f"Error: No saved bash state found for thread_id `{thread_id}`. Please re-initialize to get a new id or use correct id."
    if mode_name:
        # update modes if they're relative
        if isinstance(change_mode, CodeWriterMode):
            change_mode.update_relative_globs(starting_directory)
        else:
            assert isinstance(change_mode, str)

        # Get new state configuration
        mode_impl = modes_to_state(change_mode)

        # Reset shell with new mode, using the provided thread_id
        context.bash_state.load_state(
            mode_impl.bash_command_mode,
            mode_impl.file_edit_mode,
            mode_impl.write_if_empty_mode,
            mode_impl.mode_name,
            dict(context.bash_state.whitelist_for_overwrite),
            starting_directory,
            starting_directory,
            thread_id,
        )
        mode_prompt = get_mode_prompt(context)
        return (
            f"Reset successful with mode change to {mode_name}.\n"
            + mode_prompt
            + "\n"
            + get_status(context.bash_state, is_bg=False)
        )
    else:
        # Regular reset without mode change - keep same mode but update directory
        bash_command_mode = context.bash_state.bash_command_mode
        file_edit_mode = context.bash_state.file_edit_mode
        write_if_empty_mode = context.bash_state.write_if_empty_mode
        mode = context.bash_state.mode

        # Reload state with new directory, using the provided thread_id
        context.bash_state.load_state(
            bash_command_mode,
            file_edit_mode,
            write_if_empty_mode,
            mode,
            dict(context.bash_state.whitelist_for_overwrite),
            starting_directory,
            starting_directory,
            thread_id,
        )
    return "Reset successful" + get_status(context.bash_state, is_bg=False)


T = TypeVar("T")


def save_out_of_context(content: str, suffix: str) -> str:
    file_path = NamedTemporaryFile(delete=False, suffix=suffix).name
    with open(file_path, "w") as f:
        f.write(content)
    return file_path


def expand_user(path: str) -> str:
    if not path or not path.startswith("~"):
        return path
    return expanduser(path)


def try_open_file(file_path: str) -> None:
    """Try to open a file using the system's default application."""
    # Determine the appropriate open command based on OS
    open_cmd = None
    if os.uname().sysname == "Darwin":  # macOS
        open_cmd = "open"
    elif os.uname().sysname == "Linux":
        # Try common Linux open commands
        for cmd in ["xdg-open", "gnome-open", "kde-open"]:
            try:
                subprocess.run(["which", cmd], timeout=1, capture_output=True)
                open_cmd = cmd
                break
            except:
                continue

    # Try to open the file if a command is available
    if open_cmd:
        try:
            subprocess.run([open_cmd, file_path], timeout=2)
        except:
            pass


MEDIA_TYPES = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class ImageData(BaseModel):
    media_type: MEDIA_TYPES
    data: str

    @property
    def dataurl(self) -> str:
        return f"data:{self.media_type};base64," + self.data


Param = ParamSpec("Param")


def truncate_if_over(content: str, max_tokens: Optional[int]) -> str:
    if max_tokens and max_tokens > 0:
        tokens = default_enc.encoder(content)
        n_tokens = len(tokens)
        if n_tokens > max_tokens:
            content = (
                default_enc.decoder(tokens[: max(0, max_tokens - 100)])
                + "\n(...truncated)"
            )

    return content


def read_image_from_shell(file_path: str, context: Context) -> ImageData:
    # Expand the path before checking if it's absolute
    file_path = expand_user(file_path)

    # If not absolute after expansion, join with current working directory
    if not os.path.isabs(file_path):
        file_path = os.path.join(context.bash_state.cwd, file_path)

    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist")

    with open(file_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_type = mimetypes.guess_type(file_path)[0]
        return ImageData(media_type=image_type, data=image_b64)  # type: ignore


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
) -> tuple[
    str, dict[str, list[tuple[int, int]]]
]:  # Updated to return message and file paths with line ranges
    # Expand the path before checking if it's absolute
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
            {},  # Return empty dict instead of empty list for type consistency
        )

    error_on_exist_ = (
        error_on_exist and path_ not in context.bash_state.whitelist_for_overwrite
    )
    curr_hash = ""
    if error_on_exist and path_ in context.bash_state.whitelist_for_overwrite:
        # Ensure hash has not changed
        if os.path.exists(path_):
            with open(path_, "rb") as f:
                file_content = f.read()
                curr_hash = sha256(file_content).hexdigest()

                whitelist_data = context.bash_state.whitelist_for_overwrite[path_]

                # If we haven't fully read the file or hash has changed, require re-reading
                if curr_hash != whitelist_data.file_hash:
                    error_on_exist_ = True
                elif not whitelist_data.is_read_enough():
                    error_on_exist_ = True

    # Validate using write_if_empty_mode after checking whitelist
    allowed_globs = context.bash_state.write_if_empty_mode.allowed_globs
    if allowed_globs != "all" and not wcglob.globmatch(
        path_, allowed_globs, flags=wcglob.GLOBSTAR
    ):
        return (
            f"Error: updating file {path_} not allowed in current mode. Doesn't match allowed globs: {allowed_globs}",
            {},  # Empty dict instead of empty list
        )

    if (error_on_exist or error_on_exist_) and os.path.exists(path_):
        content = Path(path_).read_text().strip()
        if content:
            if error_on_exist_:
                file_ranges = []

                if path_ not in context.bash_state.whitelist_for_overwrite:
                    # File hasn't been read at all
                    msg = f"Error: you need to read existing file {path_} at least once before it can be overwritten.\n\n"
                    # Read the entire file
                    file_content_str, truncated, _, _, line_range = read_file(
                        path_, coding_max_tokens, noncoding_max_tokens, context, False
                    )
                    file_ranges = [line_range]

                    final_message = ""
                    if not truncated:
                        final_message = "You can now safely retry writing immediately considering the above information."

                    return (
                        (
                            msg
                            + f"Here's the existing file:\n<file-contents-numbered>\n{file_content_str}\n{final_message}\n</file-contents-numbered>"
                        ),
                        {path_: file_ranges},
                    )

                whitelist_data = context.bash_state.whitelist_for_overwrite[path_]

                if curr_hash != whitelist_data.file_hash:
                    msg = "Error: the file has changed since last read.\n\n"
                    # Read the entire file again
                    file_content_str, truncated, _, _, line_range = read_file(
                        path_, coding_max_tokens, noncoding_max_tokens, context, False
                    )
                    file_ranges = [line_range]

                    final_message = ""
                    if not truncated:
                        final_message = "You can now safely retry writing immediately considering the above information."

                    return (
                        (
                            msg
                            + f"Here's the existing file:\n<file-contents-numbered>\n{file_content_str}\n</file-contents-numbered>\n{final_message}"
                        ),
                        {path_: file_ranges},
                    )
                else:
                    # The file hasn't changed, but we haven't read enough of it
                    unread_ranges = whitelist_data.get_unread_ranges()
                    # Format the ranges as a string for display
                    ranges_str = ", ".join(
                        [f"{start}-{end}" for start, end in unread_ranges]
                    )
                    msg = f"Error: you need to read more of the file before it can be overwritten.\nUnread line ranges: {ranges_str}\n\n"

                    # Read just the unread ranges
                    paths_: list[str] = []
                    for start, end in unread_ranges:
                        paths_.append(path_ + ":" + f"{start}-{end}")
                    paths_readfiles = ReadFiles(file_paths=paths_)
                    readfiles, file_ranges_dict, truncated = read_files(
                        paths_readfiles.file_paths,
                        coding_max_tokens,
                        noncoding_max_tokens,
                        context,
                        start_line_nums=paths_readfiles.start_line_nums,
                        end_line_nums=paths_readfiles.end_line_nums,
                    )

                    final_message = ""
                    if not truncated:
                        final_message = "Now that you have read the rest of the file, you can now safely immediately retry writing but consider the new information above."

                    return (
                        (msg + "\n" + readfiles + "\n" + final_message),
                        file_ranges_dict,
                    )
    # No need to add to whitelist here - will be handled by get_tool_output

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
                check.errors,
                writefile.file_content,
                path_,
                coding_max_tokens,
                noncoding_max_tokens,
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

    # Count the lines directly from the content we're writing
    total_lines = writefile.file_content.count("\n") + 1

    return "Success" + "".join(warnings), {
        path_: [(1, total_lines)]
    }  # Return the file path with line range along with success message


def do_diff_edit(
    fedit: FileEdit,
    coding_max_tokens: Optional[int],
    noncoding_max_tokens: Optional[int],
    context: Context,
) -> tuple[str, dict[str, list[tuple[int, int]]]]:
    try:
        return _do_diff_edit(fedit, coding_max_tokens, noncoding_max_tokens, context)
    except Exception as e:
        # Try replacing \"
        try:
            fedit = FileEdit(
                file_path=fedit.file_path,
                file_edit_using_search_replace_blocks=fedit.file_edit_using_search_replace_blocks.replace(
                    '\\"', '"'
                ),
            )
            return _do_diff_edit(
                fedit, coding_max_tokens, noncoding_max_tokens, context
            )
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

    # Expand the path before checking if it's absolute
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

    # Validate using file_edit_mode
    allowed_globs = context.bash_state.file_edit_mode.allowed_globs
    if allowed_globs != "all" and not wcglob.globmatch(
        path_, allowed_globs, flags=wcglob.GLOBSTAR
    ):
        raise Exception(
            f"Error: updating file {path_} not allowed in current mode. Doesn't match allowed globs: {allowed_globs}"
        )

    # No need to add to whitelist here - will be handled by get_tool_output

    if not os.path.exists(path_):
        raise Exception(f"Error: file {path_} does not exist")

    with open(path_) as f:
        apply_diff_to = f.read()

    fedit.file_edit_using_search_replace_blocks = (
        fedit.file_edit_using_search_replace_blocks.strip()
    )
    lines = fedit.file_edit_using_search_replace_blocks.split("\n")

    apply_diff_to, comments = search_replace_edit(
        lines, apply_diff_to, context.console.log
    )

    # Count the lines just once - after the edit but before writing
    total_lines = apply_diff_to.count("\n") + 1

    with open(path_, "w") as f:
        f.write(apply_diff_to)

    syntax_errors = ""
    extension = Path(path_).suffix.lstrip(".")
    try:
        check = check_syntax(extension, apply_diff_to)
        syntax_errors = check.description
        if syntax_errors:
            context_for_errors = get_context_for_errors(
                check.errors,
                apply_diff_to,
                path_,
                coding_max_tokens,
                noncoding_max_tokens,
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
            )  # Return the file path with line range along with the warning message
    except Exception:
        pass

    return comments, {
        path_: [(1, total_lines)]
    }  # Return the file path with line range along with the edit comments


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
) -> tuple[
    str, dict[str, list[tuple[int, int]]]
]:  # Updated to return message and file paths with line ranges
    """
    Write or edit a file based on percentage of changes.
    If percentage_changed > 50%, treat content as direct file content.
    Otherwise, treat content as search/replace blocks.
    """
    # Check if the thread_id matches current
    if file_writing_args.thread_id != context.bash_state.current_thread_id:
        # Try to load state from the thread_id
        if not context.bash_state.load_state_from_thread_id(
            file_writing_args.thread_id
        ):
            return (
                f"Error: No saved bash state found for thread_id `{file_writing_args.thread_id}`. Please re-initialize to get a new id or use correct id.",
                {},
            )

    # Expand the path before checking if it's absolute
    path_ = expand_user(file_writing_args.file_path)
    if not os.path.isabs(path_):
        return (
            f"Failure: file_path should be absolute path, current working directory is {context.bash_state.cwd}",
            {},  # Return empty dict instead of empty list for type consistency
        )

    # If file doesn't exist, always use direct file_content mode
    content = file_writing_args.text_or_search_replace_blocks

    if not _is_edit(content, file_writing_args.percentage_to_change):
        # Use direct content mode (same as WriteIfEmpty)
        result, paths = write_file(
            WriteIfEmpty(
                file_path=path_,
                file_content=file_writing_args.text_or_search_replace_blocks,
            ),
            True,
            coding_max_tokens,
            noncoding_max_tokens,
            context,
        )
        return result, paths
    else:
        # File exists and percentage <= 50, use search/replace mode
        result, paths = do_diff_edit(
            FileEdit(
                file_path=path_,
                file_edit_using_search_replace_blocks=file_writing_args.text_or_search_replace_blocks,
            ),
            coding_max_tokens,
            noncoding_max_tokens,
            context,
        )
        return result, paths


TOOLS = BashCommand | FileWriteOrEdit | ReadImage | ReadFiles | Initialize | ContextSave


def which_tool(args: str) -> TOOLS:
    adapter = TypeAdapter[TOOLS](TOOLS, config={"extra": "forbid"})
    return adapter.validate_python(json.loads(args))


def which_tool_name(name: str) -> Type[TOOLS]:
    if name == "BashCommand":
        return BashCommand
    elif name == "FileWriteOrEdit":
        return FileWriteOrEdit
    elif name == "ReadImage":
        return ReadImage
    elif name == "ReadFiles":
        return ReadFiles
    elif name == "Initialize":
        return Initialize
    elif name == "ContextSave":
        return ContextSave
    else:
        raise ValueError(f"Unknown tool name: {name}")


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
    """Merge file path ranges from source into target."""
    for path, ranges in source.items():
        if path in target:
            target[path].extend(ranges)
        else:
            target[path] = list(ranges)


def _handle_initialize(
    arg: Initialize, context: Context,
    coding_max_tokens: Optional[int], noncoding_max_tokens: Optional[int],
) -> tuple[tuple[str, float], Context, dict[str, list[tuple[int, int]]]]:
    """Dispatch Initialize by subtype."""
    if arg.type in ("user_asked_mode_change", "reset_shell"):
        workspace_path = (
            arg.any_workspace_path
            if os.path.isdir(arg.any_workspace_path)
            else os.path.dirname(arg.any_workspace_path)
        )
        workspace_path = workspace_path if os.path.exists(workspace_path) else ""
        result = reset_babash(
            context, workspace_path,
            arg.mode_name if is_mode_change(arg.mode, context.bash_state) else None,
            arg.mode, arg.thread_id,
        )
        return (result, 0.0), context, {}

    init_type: Literal["user_asked_change_workspace", "first_call"] = arg.type  # type: ignore[assignment]
    output_, context, init_paths = initialize(
        init_type, context, arg.any_workspace_path,
        arg.initial_files_to_read or [], arg.task_id_to_resume,
        coding_max_tokens, noncoding_max_tokens, arg.mode, arg.thread_id,
    )
    return (output_, 0.0), context, init_paths


def _handle_context_save(arg: ContextSave, context: Context) -> str:
    """Handle ContextSave tool call."""
    relevant_files: list[str] = []
    warnings = ""
    arg.project_root_path = os.path.expanduser(arg.project_root_path)

    for fglob in arg.relevant_file_globs:
        fglob = expand_user(fglob)
        if not os.path.isabs(fglob) and arg.project_root_path:
            fglob = os.path.join(arg.project_root_path, fglob)
        globs = glob.glob(fglob, recursive=True)
        relevant_files.extend(globs[:1000])
        if not globs:
            warnings += f"Warning: No files found for the glob: {fglob}\n"

    relevant_files_data, _, _ = read_files(relevant_files[:10_000], None, None, context)
    save_path = save_memory(arg, relevant_files_data, context.bash_state.serialize())
    try_open_file(save_path)

    if not relevant_files and arg.relevant_file_globs:
        return f'Error: No files found for the given globs. Context file successfully saved at "{save_path}", but please fix the error.'
    if warnings:
        return warnings + "\nContext file successfully saved at " + save_path
    return save_path


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
        context.console.print("Calling execute bash tool")
        output_str, cost = execute_bash(
            context.bash_state, enc, arg, noncoding_max_tokens, arg.action_json.wait_for_seconds,
        )
        output = output_str, cost

    elif isinstance(arg, WriteIfEmpty):
        context.console.print("Calling write file tool")
        result, paths = write_file(arg, True, coding_max_tokens, noncoding_max_tokens, context)
        output = result, 0.0
        _merge_ranges(file_paths_with_ranges, paths)

    elif isinstance(arg, FileEdit):
        context.console.print("Calling full file edit tool")
        result, paths = do_diff_edit(arg, coding_max_tokens, noncoding_max_tokens, context)
        output = result, 0.0
        _merge_ranges(file_paths_with_ranges, paths)

    elif isinstance(arg, FileWriteOrEdit):
        context.console.print("Calling file writing tool")
        result, paths = file_writing(arg, coding_max_tokens, noncoding_max_tokens, context)
        output = result, 0.0
        _merge_ranges(file_paths_with_ranges, paths)

    elif isinstance(arg, ReadImage):
        context.console.print("Calling read image tool")
        output = read_image_from_shell(arg.file_path, context), 0.0

    elif isinstance(arg, ReadFiles):
        context.console.print("Calling read file tool")
        result, paths, _ = read_files(
            arg.file_paths, coding_max_tokens, noncoding_max_tokens, context,
            arg.start_line_nums, arg.end_line_nums,
        )
        output = result, 0.0
        _merge_ranges(file_paths_with_ranges, paths)

    elif isinstance(arg, Initialize):
        context.console.print("Calling initial info tool")
        output, context, init_paths = _handle_initialize(
            arg, context, coding_max_tokens, noncoding_max_tokens
        )
        _merge_ranges(file_paths_with_ranges, {
            p: r for p, r in init_paths.items() if os.path.exists(p)
        })

    elif isinstance(arg, ContextSave):
        context.console.print("Calling task memory tool")
        output = _handle_context_save(arg, context), 0.0

    else:
        raise ValueError(f"Unknown tool: {arg}")

    if file_paths_with_ranges:
        context.bash_state.add_to_whitelist_for_overwrite(file_paths_with_ranges)
    context.bash_state.save_state_to_disk()

    if isinstance(output[0], str):
        context.console.print(str(output[0]))
    else:
        context.console.print(f"Received {type(output[0])} from tool")
    return [output[0]], output[1]



default_enc = get_default_encoder()
curr_cost = 0.0


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
) -> tuple[
    str, dict[str, list[tuple[int, int]]], bool
]:  # Updated to return file paths with ranges
    message = ""
    file_ranges_dict: dict[
        str, list[tuple[int, int]]
    ] = {}  # Map file paths to line ranges

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
            # Use line numbers from parameters if provided
            start_line_num = None if start_line_nums is None else start_line_nums[i]
            end_line_num = None if end_line_nums is None else end_line_nums[i]

            # For backward compatibility, we still need to extract line numbers from path
            # if they weren't provided as parameters
            content, truncated, tokens, path, line_range = read_file(
                file,
                coding_max_tokens,
                noncoding_max_tokens,
                context,
                start_line_num,
                end_line_num,
            )

            # Add file path with line range to dictionary
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

        # Check if we've hit both token limit
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
    show_line_numbers = True
    # Line numbers are now passed as parameters, no need to parse from path

    # Expand the path before checking if it's absolute
    file_path = expand_user(file_path)

    if not os.path.isabs(file_path):
        raise ValueError(
            f"Failure: file_path should be absolute path, current working directory is {context.bash_state.cwd}"
        )

    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"Error: file {file_path} does not exist")

    # Read all lines of the file
    with path.open("r") as f:
        all_lines = f.readlines(10_000_000)

        if all_lines and all_lines[-1].endswith("\n"):
            # Special handling of line counts because readlines doesn't consider last empty line as a separate line
            all_lines.append("")

    total_lines = len(all_lines)

    # Apply line range filtering if specified
    start_idx = 0
    if start_line_num is not None:
        # Convert 1-indexed line number to 0-indexed
        start_idx = max(0, start_line_num - 1)

    end_idx = len(all_lines)
    if end_line_num is not None:
        # end_line_num is inclusive, so we use min to ensure it's within bounds
        end_idx = min(len(all_lines), end_line_num)

    # Convert back to 1-indexed line numbers for tracking
    effective_start = start_line_num if start_line_num is not None else 1
    effective_end = end_line_num if end_line_num is not None else total_lines

    filtered_lines = all_lines[start_idx:end_idx]

    # Create content with or without line numbers
    if show_line_numbers:
        content_lines = []
        for i, line in enumerate(filtered_lines, start=start_idx + 1):
            content_lines.append(f"{i} {line}")
        content = "".join(content_lines)
    else:
        content = "".join(filtered_lines)

    truncated = False
    tokens_counts = 0

    # Select the appropriate max_tokens based on file type
    max_tokens = select_max_tokens(file_path, coding_max_tokens, noncoding_max_tokens)

    # Handle token limit if specified
    if max_tokens is not None:
        tokens = default_enc.encoder(content)
        tokens_counts = len(tokens)

        if len(tokens) > max_tokens:
            # Truncate at token boundary first
            truncated_tokens = tokens[:max_tokens]
            truncated_content = default_enc.decoder(truncated_tokens)

            # Count how many lines we kept
            line_count = truncated_content.count("\n")

            # Calculate the last line number shown (1-indexed)
            last_line_shown = start_idx + line_count

            content = truncated_content
            # Add informative message about truncation with total line count
            total_lines = len(all_lines)
            content += (
                f"\n(...truncated) Only showing till line number {last_line_shown} of {total_lines} total lines due to the token limit, please continue reading from {last_line_shown + 1} if required"
                f" using syntax {file_path}:{last_line_shown + 1}-{total_lines}"
            )
            truncated = True

            # Update effective_end if truncated
            effective_end = last_line_shown

    # Return the content along with the effective line range that was read
    return (
        content,
        truncated,
        tokens_counts,
        file_path,
        (effective_start, effective_end),
    )


if __name__ == "__main__":
    with BashState(
        rich.console.Console(style="blue", highlight=False, markup=False),
        "",
        None,
        None,
        None,
        None,
        True,
        None,
        None,
        None,
    ) as BASH_STATE:
        print(
            get_tool_output(
                Context(BASH_STATE, BASH_STATE.console),
                Initialize(
                    type="first_call",
                    any_workspace_path="",
                    initial_files_to_read=[],
                    task_id_to_resume="",
                    mode_name="babash",
                    thread_id="",
                ),
                default_enc,
                0,
                lambda x, y: ("", 0),
                24000,  # coding_max_tokens
                8000,  # noncoding_max_tokens
            )
        )
        print(
            get_tool_output(
                Context(BASH_STATE, BASH_STATE.console),
                BashCommand(
                    action_json=Command(
                        command="pwd", thread_id=BASH_STATE.current_thread_id
                    )
                ),
                default_enc,
                0,
                lambda x, y: ("", 0),
                24000,  # coding_max_tokens
                8000,  # noncoding_max_tokens
            )
        )

        print(
            get_tool_output(
                Context(BASH_STATE, BASH_STATE.console),
                BashCommand(
                    action_json=Command(
                        command="source .venv/bin/activate",
                        thread_id=BASH_STATE.current_thread_id,
                    )
                ),
                default_enc,
                0,
                lambda x, y: ("", 0),
                24000,  # coding_max_tokens
                8000,  # noncoding_max_tokens
            )
        )

        print(
            get_tool_output(
                Context(BASH_STATE, BASH_STATE.console),
                BashCommand(
                    action_json=Command(
                        command="pwd", thread_id=BASH_STATE.current_thread_id
                    )
                ),
                default_enc,
                0,
                lambda x, y: ("", 0),
                24000,  # coding_max_tokens
                8000,  # noncoding_max_tokens
            )
        )

        print(
            get_tool_output(
                Context(BASH_STATE, BASH_STATE.console),
                BashCommand(
                    action_json=Command(
                        command="take src", thread_id=BASH_STATE.current_thread_id
                    )
                ),
                default_enc,
                0,
                lambda x, y: ("", 0),
                24000,  # coding_max_tokens
                8000,  # noncoding_max_tokens
            )
        )

        print(
            get_tool_output(
                Context(BASH_STATE, BASH_STATE.console),
                BashCommand(
                    action_json=Command(
                        command="pwd", thread_id=BASH_STATE.current_thread_id
                    )
                ),
                default_enc,
                0,
                lambda x, y: ("", 0),
                24000,  # coding_max_tokens
                8000,  # noncoding_max_tokens
            )
        )
