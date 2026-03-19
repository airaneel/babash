"""Tests for extracted helper functions in tools.py."""

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from babash.client.tools import (
    _load_alignment_docs,
    _resolve_workspace,
    _resume_task,
    expand_user,
    save_out_of_context,
    truncate_if_over,
    default_enc,
)
from babash.types_ import _patch_singleton_all


def test_patch_singleton_all() -> None:
    assert _patch_singleton_all(["all"]) == "all"
    assert _patch_singleton_all(["src/**"]) == ["src/**"]
    assert _patch_singleton_all("all") == "all"
    assert _patch_singleton_all([]) == []


def test_expand_user() -> None:
    assert expand_user("") == ""
    assert expand_user("/absolute/path") == "/absolute/path"
    home = os.path.expanduser("~")
    assert expand_user("~/test").startswith(home)


def test_truncate_if_over() -> None:
    short = "hello world"
    assert truncate_if_over(short, 1000) == short
    assert truncate_if_over(short, None) == short

    long_text = "word " * 10000
    truncated = truncate_if_over(long_text, 100)
    assert "truncated" in truncated
    assert len(truncated) < len(long_text)


def test_save_out_of_context() -> None:
    path = save_out_of_context("test content", ".txt")
    assert os.path.exists(path)
    with open(path) as f:
        assert f.read() == "test content"
    os.unlink(path)


def test_resume_task_empty() -> None:
    memory, workspace, state = _resume_task("", True, None, None)
    assert memory == ""
    assert workspace == ""
    assert state is None


def test_resume_task_not_first_call() -> None:
    memory, workspace, state = _resume_task("some-id", False, None, None)
    assert "Warning" in memory
    assert state is None


def test_resolve_workspace_empty() -> None:
    ctx, folder, files = _resolve_workspace("", True, [], "babash")
    assert folder is None or isinstance(folder, Path)
    # Should create a playground dir
    assert ctx == "" or "claude-playground" in ctx or folder is not None


def test_resolve_workspace_existing_dir(tmp_path: Any) -> None:
    ctx, folder, files = _resolve_workspace(str(tmp_path), False, [], "babash")
    assert folder is not None or ctx != ""


def test_resolve_workspace_nonexistent() -> None:
    with tempfile.TemporaryDirectory() as td:
        nonexistent = os.path.join(td, "subdir")
        ctx, folder, files = _resolve_workspace(nonexistent, False, [], "babash")
        assert folder is not None
        assert os.path.exists(nonexistent)  # should have been created


def test_resolve_workspace_file(tmp_path: Any) -> None:
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")
    ctx, folder, files = _resolve_workspace(str(test_file), False, [], "babash")
    assert str(test_file) in files


def test_load_alignment_docs_empty() -> None:
    console = MagicMock()
    result = _load_alignment_docs(None, console)
    assert isinstance(result, str)


def test_load_alignment_docs_with_claude_md(tmp_path: Any) -> None:
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("# Test instructions")
    console = MagicMock()
    result = _load_alignment_docs(tmp_path, console)
    assert "Test instructions" in result
    assert "guidelines" in result.lower()


def test_load_alignment_docs_agents_md_fallback(tmp_path: Any) -> None:
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text("# Agent rules")
    console = MagicMock()
    result = _load_alignment_docs(tmp_path, console)
    assert "Agent rules" in result
