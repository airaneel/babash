import os
import subprocess
from collections import deque
from pathlib import Path
from typing import Optional

from .display_tree import DirectoryTree
from .file_stats import load_workspace_stats
from .path_score import sort_paths_by_importance

MAX_ENTRIES_CHECK = 100_000
# How many commits to look back through when collecting recently-touched files.
RECENT_COMMITS_SCANNED = 100


def _git(args: list[str], cwd: str) -> Optional[str]:
    """Run a git command and return stdout, or None if git is unavailable or the
    command fails (not a repo, no commits yet, …). Never raises."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def find_repo_root(path: Path) -> Optional[Path]:
    """Repo root containing `path`, or None if it isn't inside a git repo."""
    start = path.parent if path.is_file() else path
    if not start.is_dir():
        return None
    out = _git(["rev-parse", "--show-toplevel"], str(start))
    if not out or not out.strip():
        return None
    return Path(out.strip())


def get_git_files(repo_root: Path) -> list[str]:
    """Every file git would show: tracked + untracked, minus ignored.

    One `git ls-files` call replaces a directory walk that used to ask pygit2
    `path_is_ignored()` once per entry.
    """
    out = _git(["ls-files", "--cached", "--others", "--exclude-standard"], str(repo_root))
    if out is None:
        return []
    files = [line for line in out.splitlines() if line]
    return files[:MAX_ENTRIES_CHECK]


def get_all_files_max_depth(abs_folder: str, max_depth: int) -> list[str]:
    """BFS over a plain (non-git) directory, returning relative paths."""
    all_files: list[str] = []
    queue = deque([(abs_folder, 0, "")])
    entries_check = 0
    while queue and entries_check < MAX_ENTRIES_CHECK:
        current_folder, depth, prefix = queue.popleft()

        if depth > max_depth:
            continue

        try:
            entries = list(os.scandir(current_folder))
        except OSError:
            continue

        files: list[str] = []
        folders: list[tuple[str, str]] = []
        for entry in entries:
            entries_check += 1
            try:
                is_file = entry.is_file(follow_symlinks=False)
            except OSError:
                continue
            rel_path = f"{prefix}{entry.name}" if prefix else entry.name
            if is_file:
                files.append(rel_path)
            else:
                folders.append((entry.path, rel_path))

        chunk = files[: min(10_000, max(0, MAX_ENTRIES_CHECK - entries_check))]
        all_files.extend(chunk)

        for folder_path, folder_rel_path in folders:
            queue.append((folder_path, depth + 1, f"{folder_rel_path}/"))

    return all_files


def get_recent_git_files(repo_root: Path, count: int) -> list[str]:
    """Files touched by the most recent commits, newest first.

    `git log --name-only` gives this directly; the previous implementation
    computed a full diff per commit just to read the changed file names.
    """
    out = _git(
        [
            "log",
            "--no-merges",
            "--name-only",
            "--pretty=format:",
            "-n",
            str(RECENT_COMMITS_SCANNED),
        ],
        str(repo_root),
    )
    if out is None:
        return []

    seen: set[str] = set()
    recent: list[str] = []
    for line in out.splitlines():
        file_path = line.strip()
        if not file_path or file_path in seen:
            continue
        # Skip files deleted since (the log still lists them).
        if not (repo_root / file_path).exists():
            continue
        seen.add(file_path)
        recent.append(file_path)
        if len(recent) >= count:
            break
    return recent


def calculate_dynamic_file_limit(total_files: int) -> int:
    min_files = 50
    max_files = 400

    if total_files <= min_files:
        return min_files

    scale_factor = (max_files - min_files) / (30000 - min_files)
    dynamic_limit = min_files + int((total_files - min_files) * scale_factor)
    return min(max_files, dynamic_limit)


def _active_files(workspace_stats_path: str, all_files: list[str], context_dir: Path) -> list[str]:
    """Top files by recorded read/edit/write activity, most active first."""
    workspace_stats = load_workspace_stats(workspace_stats_path)

    scored: list[tuple[str, int]] = []
    for file_path, file_stats in workspace_stats.files.items():
        try:
            if str(context_dir) in file_path:
                rel_path = os.path.relpath(file_path, str(context_dir))
            else:
                rel_path = file_path
            activity = (
                file_stats.read_count * 2 + file_stats.edit_count + file_stats.write_count
            )
            if rel_path in all_files or os.path.exists(file_path):
                scored.append((rel_path, activity))
        except (ValueError, OSError):
            continue

    return [f for f, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:5]]


def get_repo_context(file_or_repo_path: str) -> tuple[str, Path]:
    path = Path(file_or_repo_path).absolute()

    repo_root = find_repo_root(path)

    if repo_root is not None:
        context_dir = repo_root
        all_files = get_git_files(repo_root)
    else:
        context_dir = path.parent if path.is_file() else path
        all_files = get_all_files_max_depth(str(context_dir), 10)

    if repo_root is not None:
        dynamic_max_files = calculate_dynamic_file_limit(len(all_files))
        recent_files_count = max(10, int(dynamic_max_files * 0.2))
        recent_git_files = get_recent_git_files(repo_root, recent_files_count)
    else:
        # No dynamic limit for plain folders like /tmp or ~.
        dynamic_max_files = 50
        recent_git_files = []

    known = set(all_files)
    top_files: list[str] = []

    # The two signals that actually carry information come first: what this
    # workspace has been reading/editing, then what git touched recently.
    for file in _active_files(str(context_dir), all_files, context_dir):
        if file not in top_files and file in known:
            top_files.append(file)

    for file in recent_git_files:
        if file not in top_files and file in known:
            top_files.append(file)

    # Heuristic ordering fills the remainder.
    if len(top_files) < dynamic_max_files:
        for file in sort_paths_by_importance(all_files):
            if len(top_files) >= dynamic_max_files:
                break
            if file not in top_files:
                top_files.append(file)

    directory_printer = DirectoryTree(context_dir, max_files=dynamic_max_files)
    for file in top_files[:dynamic_max_files]:
        directory_printer.expand(file)

    return directory_printer.display(), context_dir
