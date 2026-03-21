"""Initialize and mode management operations."""

import glob
import os
import subprocess
import traceback
import uuid
from os.path import expanduser
from pathlib import Path
from typing import Any, Literal, Optional

from ...types_ import (
    CodeWriterMode,
    ContextSave,
    Initialize,
    Modes,
    ModesConfig,
)
from ..bash_state import BashState, generate_thread_id, get_status, get_tmpdir
from ..memory import load_memory, save_memory
from ..modes import ARCHITECT_PROMPT, BABASH_PROMPT, code_writer_prompt, modes_to_state
from ..repo_ops.repo_context import get_repo_context
from .context import Context, default_enc, expand_user
from .read_ops import read_files


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


def is_mode_change(mode_config: ModesConfig, bash_state: BashState) -> bool:
    mode_impl = modes_to_state(mode_config)
    return (
        mode_impl.bash_command_mode != bash_state.bash_command_mode
        or mode_impl.file_edit_mode != bash_state.file_edit_mode
        or mode_impl.write_if_empty_mode != bash_state.write_if_empty_mode
        or mode_impl.mode_name != bash_state.mode
    )


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

    if type != "first_call" and thread_id and thread_id != context.bash_state.current_thread_id:
        if not context.bash_state.load_state_from_thread_id(thread_id):
            return (
                f"Error: No saved bash state found for thread_id `{thread_id}`. Please re-initialize.",
                context, {},
            )

    memory, workspace_override, loaded_state = _resume_task(
        task_id_to_resume, type == "first_call", coding_max_tokens, noncoding_max_tokens
    )
    if workspace_override:
        any_workspace_path = workspace_override

    repo_context, folder_to_start, read_files_ = _resolve_workspace(
        any_workspace_path, type == "first_call", read_files_, mode
    )

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


def reset_babash(
    context: Context,
    starting_directory: str,
    mode_name: Optional[Modes],
    change_mode: ModesConfig,
    thread_id: str,
) -> str:
    if thread_id and thread_id != context.bash_state.current_thread_id:
        if not context.bash_state.load_state_from_thread_id(thread_id):
            return f"Error: No saved bash state found for thread_id `{thread_id}`."

    if mode_name:
        if isinstance(change_mode, CodeWriterMode):
            change_mode.update_relative_globs(starting_directory)

        mode_impl = modes_to_state(change_mode)
        context.bash_state.load_state(
            mode_impl.bash_command_mode, mode_impl.file_edit_mode, mode_impl.write_if_empty_mode,
            mode_impl.mode_name, dict(context.bash_state.whitelist_for_overwrite),
            starting_directory, starting_directory, thread_id or context.bash_state.current_thread_id,
        )
        mode_prompt = get_mode_prompt(context)
        return (
            f"Reset successful with mode change to {mode_name}.\n"
            + mode_prompt + "\n"
            + get_status(context.bash_state, is_bg=False)
        )

    bash_command_mode = context.bash_state.bash_command_mode
    file_edit_mode = context.bash_state.file_edit_mode
    write_if_empty_mode = context.bash_state.write_if_empty_mode
    mode = context.bash_state.mode

    context.bash_state.load_state(
        bash_command_mode, file_edit_mode, write_if_empty_mode, mode,
        dict(context.bash_state.whitelist_for_overwrite),
        starting_directory, starting_directory, thread_id or context.bash_state.current_thread_id,
    )
    return "Reset successful" + get_status(context.bash_state, is_bg=False)


def _handle_initialize(
    arg: Initialize, context: Context,
    coding_max_tokens: Optional[int], noncoding_max_tokens: Optional[int],
) -> tuple[tuple[str, float], Context, dict[str, list[tuple[int, int]]]]:
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


def try_open_file(file_path: str) -> None:
    open_cmd = None
    if os.uname().sysname == "Darwin":
        open_cmd = "open"
    elif os.uname().sysname == "Linux":
        for cmd in ["xdg-open", "gnome-open", "kde-open"]:
            try:
                subprocess.run(["which", cmd], timeout=1, capture_output=True)
                open_cmd = cmd
                break
            except Exception:
                continue
    if open_cmd:
        try:
            subprocess.run([open_cmd, file_path], timeout=2)
        except Exception:
            pass
