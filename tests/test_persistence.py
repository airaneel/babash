"""Tests for bash state persistence and utility functions."""

import os
from typing import Any

from babash.client.bash_state.persistence import (
    generate_thread_id,
    load_bash_state_by_id,
    save_bash_state_by_id,
)
from babash.client.bash_state.file_whitelist import FileWhitelistData
from babash.client.file_ops.extensions import (
    get_context_length_for_file,
    is_source_code_file,
    select_max_tokens,
)
from babash.types_ import ReadFiles


def test_generate_thread_id() -> None:
    tid = generate_thread_id()
    assert tid.startswith("i")
    # Collision-free id (uuid-based): word chars, and distinct across calls —
    # this uniqueness is what keeps concurrent shells from clobbering each
    # other's on-disk state.
    assert tid[1:].isalnum()
    assert len(tid) > 5
    assert generate_thread_id() != generate_thread_id()


def test_save_and_load_bash_state(tmp_path: Any) -> None:
    os.environ["XDG_DATA_HOME"] = str(tmp_path)
    try:
        state = {"mode": "babash", "workspace_root": "/tmp"}
        save_bash_state_by_id("test123", state)
        loaded = load_bash_state_by_id("test123")
        assert loaded == state
    finally:
        del os.environ["XDG_DATA_HOME"]


def test_save_empty_thread_id(tmp_path: Any) -> None:
    os.environ["XDG_DATA_HOME"] = str(tmp_path)
    try:
        save_bash_state_by_id("", {"key": "val"})
        # Should not create any file
        assert load_bash_state_by_id("") is None
    finally:
        del os.environ["XDG_DATA_HOME"]


def test_load_nonexistent(tmp_path: Any) -> None:
    os.environ["XDG_DATA_HOME"] = str(tmp_path)
    try:
        assert load_bash_state_by_id("nonexistent") is None
    finally:
        del os.environ["XDG_DATA_HOME"]


def test_whitelist_zero_lines() -> None:
    w = FileWhitelistData(file_hash="abc", line_ranges_read=[], total_lines=0)
    assert w.get_percentage_read() == 100.0
    assert w.is_read_enough()
    assert w.get_unread_ranges() == []


def test_whitelist_serialize_roundtrip() -> None:
    w = FileWhitelistData(file_hash="abc", line_ranges_read=[(1, 10)], total_lines=20)
    data = w.serialize()
    w2 = FileWhitelistData.deserialize(data)
    assert w2.file_hash == "abc"
    assert w2.line_ranges_read == [(1, 10)]
    assert w2.total_lines == 20


def test_is_source_code_file() -> None:
    assert is_source_code_file("main.py")
    assert is_source_code_file("app.tsx")
    assert is_source_code_file("Makefile")  # matched via extensionless fallback
    assert is_source_code_file("data.csv") is False


def test_get_context_length_for_file() -> None:
    code_len = get_context_length_for_file("main.py")
    default_len = get_context_length_for_file("readme.txt")
    assert code_len > 0
    assert default_len > 0


def test_select_max_tokens() -> None:
    assert select_max_tokens("main.py", 24000, 8000) == 24000
    assert select_max_tokens("data.txt", 24000, 8000) == 8000
    assert select_max_tokens("any.py", None, None) is None


def test_readfiles_line_range_parsing() -> None:
    # Basic line number
    rf = ReadFiles(file_paths=["file.py:10"])
    assert rf.file_paths == ["file.py"]
    assert rf.start_line_nums == [10]
    assert rf.end_line_nums == [None]

    # Range
    rf = ReadFiles(file_paths=["file.py:10-20"])
    assert rf.file_paths == ["file.py"]
    assert rf.start_line_nums == [10]
    assert rf.end_line_nums == [20]

    # Open-ended start
    rf = ReadFiles(file_paths=["file.py:-20"])
    assert rf.file_paths == ["file.py"]
    assert rf.start_line_nums == [None]
    assert rf.end_line_nums == [20]

    # Open-ended end
    rf = ReadFiles(file_paths=["file.py:10-"])
    assert rf.file_paths == ["file.py"]
    assert rf.start_line_nums == [10]
    assert rf.end_line_nums == [None]

    # No range
    rf = ReadFiles(file_paths=["/path/to/file.py"])
    assert rf.file_paths == ["/path/to/file.py"]
    assert rf.start_line_nums == [None]
    assert rf.end_line_nums == [None]

    # Multiple files mixed
    rf = ReadFiles(file_paths=["a.py:1-5", "b.py", "c.py:10"])
    assert rf.file_paths == ["a.py", "b.py", "c.py"]
    assert rf.start_line_nums == [1, None, 10]
    assert rf.end_line_nums == [5, None, None]
