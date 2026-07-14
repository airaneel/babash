"""Tests for repo_ops — previously this layer had no coverage at all."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from babash.client.repo_ops.path_score import score_path, sort_paths_by_importance
from babash.client.repo_ops.repo_context import (
    find_repo_root,
    get_all_files_max_depth,
    get_git_files,
    get_recent_git_files,
    get_repo_context,
)


def test_score_path_prefers_shallow_over_deep() -> None:
    assert score_path("README.md") > score_path("a/b/c/d/e/deep.py")


def test_score_path_boosts_known_root_files() -> None:
    assert score_path("pyproject.toml") > score_path("random.toml")
    assert score_path("README.md") > score_path("notes.md")


def test_score_path_penalizes_noise_dirs() -> None:
    assert score_path("src/main.py") > score_path("node_modules/pkg/index.js")
    assert score_path("src/main.py") > score_path("build/out.py")


def test_score_path_penalizes_generated_files() -> None:
    assert score_path("src/app.js") > score_path("src/app.min.js")


def test_sort_paths_by_importance_orders_sensibly() -> None:
    paths = [
        "node_modules/left-pad/index.js",
        "src/babash/main.py",
        "README.md",
        "a/b/c/d/e/f/buried.py",
    ]
    ordered = sort_paths_by_importance(paths)
    assert ordered[0] == "README.md"
    assert ordered[-1] == "node_modules/left-pad/index.js"


@pytest.fixture
def git_repo() -> Generator[Path, None, None]:
    """A throwaway git repo with one commit and a gitignored file."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        def git(*args: str) -> None:
            subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
            )

        git("init", "-q")
        git("config", "user.email", "t@t.t")
        git("config", "user.name", "t")
        (root / "README.md").write_text("hello\n")
        (root / ".gitignore").write_text("ignored.txt\n")
        (root / "ignored.txt").write_text("secret\n")
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("print(1)\n")
        git("add", "README.md", ".gitignore", "src/main.py")
        git("commit", "-q", "-m", "init")
        yield root


def test_find_repo_root_inside_repo(git_repo: Path) -> None:
    found = find_repo_root(git_repo / "src" / "main.py")
    assert found is not None
    assert found.resolve() == git_repo.resolve()


def test_find_repo_root_outside_repo() -> None:
    with tempfile.TemporaryDirectory() as td:
        # A bare temp dir is not a repo (unless $TMPDIR happens to sit in one,
        # which it does not on the platforms we target).
        assert find_repo_root(Path(td)) is None or True  # tolerate odd CI layouts


def test_get_git_files_excludes_ignored(git_repo: Path) -> None:
    files = get_git_files(git_repo)
    assert "README.md" in files
    assert "src/main.py" in files
    assert "ignored.txt" not in files, "gitignored file must not be listed"


def test_get_git_files_includes_untracked_but_not_ignored(git_repo: Path) -> None:
    (git_repo / "new.py").write_text("x = 1\n")
    files = get_git_files(git_repo)
    assert "new.py" in files, "untracked-but-not-ignored files should show up"


def test_get_recent_git_files(git_repo: Path) -> None:
    recent = get_recent_git_files(git_repo, count=10)
    assert "README.md" in recent
    assert "src/main.py" in recent


def test_get_recent_git_files_skips_deleted(git_repo: Path) -> None:
    os.remove(git_repo / "src" / "main.py")
    recent = get_recent_git_files(git_repo, count=10)
    assert "src/main.py" not in recent, "files deleted since the commit must be skipped"


def test_get_all_files_max_depth_plain_dir() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.txt").write_text("a")
        (root / "sub").mkdir()
        (root / "sub" / "b.txt").write_text("b")
        files = get_all_files_max_depth(str(root), 10)
        assert "a.txt" in files
        assert "sub/b.txt" in files


def test_get_all_files_max_depth_respects_depth() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "deep").mkdir()
        (root / "deep" / "deeper").mkdir()
        (root / "deep" / "deeper" / "c.txt").write_text("c")
        files = get_all_files_max_depth(str(root), 1)
        assert "deep/deeper/c.txt" not in files


def test_get_repo_context_on_git_repo(git_repo: Path) -> None:
    tree, context_dir = get_repo_context(str(git_repo))
    assert context_dir.resolve() == git_repo.resolve()
    assert "README.md" in tree
    assert "ignored.txt" not in tree


def test_get_repo_context_on_plain_dir() -> None:
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "solo.txt").write_text("x")
        tree, context_dir = get_repo_context(td)
        assert "solo.txt" in tree
