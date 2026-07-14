"""Regression tests for bugs fixed in the file_ops search/replace engine.

Each test here failed before the corresponding fix.
"""

from babash.client.file_ops.diff_edit import fix_indentation, match_exact
from babash.client.file_ops.search_replace import search_replace_edit


def _noop_logger(msg: str) -> None:
    pass


def test_match_exact_finds_block_at_end_of_file_with_offset() -> None:
    """content_offset must not shrink the searched range.

    `n_content` is a *count* of remaining lines, but was being used as the loop's
    end index, so with content_offset > 0 the last `content_offset` lines were
    never indexed — a search block near EOF silently failed to match.
    """
    content = [f"line{i}" for i in range(10)]
    # Search for the final line, starting the scan at offset 5.
    matches = match_exact(content, content_offset=5, search=["line9"])
    assert matches, "block at end of file must be found when content_offset > 0"
    assert matches[0].start == 9


def test_second_block_can_match_at_end_of_file() -> None:
    """End-to-end version: the 2nd block sits at EOF, so it is searched with a
    non-zero offset left by the 1st block."""
    original = "alpha\nbeta\ngamma\ndelta\nepsilon\n"
    blocks = (
        "<<<<<<< SEARCH\n"
        "alpha\n"
        "=======\n"
        "ALPHA\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "epsilon\n"
        "=======\n"
        "EPSILON\n"
        ">>>>>>> REPLACE\n"
    )
    edited, _ = search_replace_edit(blocks.split("\n"), original, _noop_logger)
    assert "ALPHA" in edited
    assert "EPSILON" in edited, "block at EOF must still be replaceable"


def test_fix_indentation_all_blank_lines_does_not_raise() -> None:
    """All-blank matched/searched lines produce empty indent lists, so `diffs`
    is empty and `diffs[0]` used to raise IndexError."""
    replaced = ["    body"]
    assert fix_indentation(["", "  "], ["", "\t"], replaced) == replaced


def test_fix_indentation_reindents() -> None:
    # searched is indented 2 more than matched -> replacement loses 2 spaces.
    out = fix_indentation(["foo"], ["  foo"], ["    bar"])
    assert out == ["  bar"]
