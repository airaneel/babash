"""Tests for the incremental terminal renderer and the prompt sentinel.

The renderer replaced a version that rebuilt a pyte Screen and re-fed the whole
buffer on every poll. These pin down both the bug that design carried and the
behaviour it did get right.
"""

import os
import tempfile

import pyte
import pyte.modes as pyte_modes

from babash.client.bash_state.shell_process import (
    MARKER_END,
    MARKER_START,
    PROMPT_CONST,
    TerminalRenderer,
    babash_rc_block,
    ensure_babash_block_in_rc_file,
)


class _Console:
    def print(self, *args: object, **kwargs: object) -> None:
        pass

    def log(self, *args: object, **kwargs: object) -> None:
        pass


def _render_from_scratch(text: str) -> list[str]:
    """What the old renderer did: a fresh Screen fed the entire buffer."""
    screen = pyte.Screen(160, 500)
    screen.set_mode(pyte_modes.LNM)
    pyte.Stream(screen).feed(text)
    dsp = screen.display[::-1]
    for i, line in enumerate(dsp):
        if line.strip():
            break
    else:
        i = len(dsp)
    return screen.display[: len(dsp) - i]


def test_output_past_100kb_is_still_delivered() -> None:
    """The regression this renderer exists for.

    The old code capped its input at the last 100_000 chars to bound the cost of
    re-rendering, but computed "what's new" using an offset measured against the
    *uncapped* buffer. Past 100KB the two disagreed and every later poll reported
    nothing at all — a command's output silently went dark partway through.
    """
    renderer = TerminalRenderer()
    buffer = ""
    for i in range(4000):  # ~120KB, well past the old cliff
        buffer += f"line {i} of output padded out to some width\r\n"
        if i % 500 == 0:
            renderer.incremental(buffer)

    buffer += "THE-VERY-LAST-LINE\r\n"
    assert "THE-VERY-LAST-LINE" in renderer.incremental(buffer)


def test_incremental_feed_matches_full_render() -> None:
    """Feeding only new bytes must land on the same screen as re-feeding everything."""
    renderer = TerminalRenderer()
    buffer = ""
    for i in range(50):
        buffer += f"line {i}\r\n"
        renderer.incremental(buffer)

    assert renderer._display() == _render_from_scratch(buffer)


def test_no_new_bytes_means_no_new_output() -> None:
    renderer = TerminalRenderer()
    buffer = "hello\r\nworld\r\n"
    assert "world" in renderer.incremental(buffer)
    assert renderer.incremental(buffer) == ""


def test_redrawn_last_line_is_reemitted() -> None:
    """A line rewritten in place (a progress bar) must be re-sent, not treated as
    already seen just because we reported an earlier version of it."""
    renderer = TerminalRenderer()
    assert "50%" in renderer.incremental("done\r\nprogress: 50%")
    assert "99%" in renderer.incremental("done\r\nprogress: 50%\rprogress: 99%")


def test_reset_starts_a_clean_screen() -> None:
    renderer = TerminalRenderer()
    renderer.incremental("first command output\r\n")
    renderer.reset()
    out = renderer.incremental("second command output\r\n")
    assert "second" in out
    assert "first" not in out


def test_rewound_buffer_rebuilds_rather_than_feeding_a_bogus_delta() -> None:
    renderer = TerminalRenderer()
    renderer.incremental("a very long first buffer\r\n" * 10)
    assert "short" in renderer.incremental("short\r\n")


def test_prompt_matches_exit_code_and_cwd() -> None:
    match = PROMPT_CONST.search("◉ 42|/home/user/some dir──➤")
    assert match is not None
    assert match.group(1) == "42"
    assert match.group(2) == "/home/user/some dir"


def test_prompt_cwd_may_contain_the_separator() -> None:
    match = PROMPT_CONST.search("◉ 0|/tmp/we|rd──➤")
    assert match is not None
    assert match.group(1) == "0"
    assert match.group(2) == "/tmp/we|rd"


def test_rc_block_is_rewritten_when_stale() -> None:
    """An older babash pinned an older prompt in the rc file. Leaving that block
    alone because its marker is present would mean the new PROMPT_CONST never
    matches the shell it starts."""
    with tempfile.TemporaryDirectory() as td:
        rc = os.path.join(td, ".bashrc")
        stale = f"{MARKER_START}\nPROMPT_COMMAND='printf \"◉ $(pwd)──➤\"'\n{MARKER_END}"
        with open(rc, "w") as f:
            f.write(f"export FOO=1\n{stale}\nexport BAR=2\n")

        home = os.environ.get("HOME")
        os.environ["HOME"] = td
        try:
            ensure_babash_block_in_rc_file("/bin/bash", _Console())
            with open(rc) as f:
                content = f.read()
        finally:
            if home is not None:
                os.environ["HOME"] = home

        expected = babash_rc_block("bash")
        assert expected is not None
        assert expected.strip() in content
        assert "◉ $(pwd)──➤" not in content, "stale prompt must be gone"
        # The user's own lines survive.
        assert "export FOO=1" in content
        assert "export BAR=2" in content
        assert content.count(MARKER_START) == 1


def test_rc_block_is_left_alone_when_current() -> None:
    with tempfile.TemporaryDirectory() as td:
        rc = os.path.join(td, ".bashrc")
        block = babash_rc_block("bash")
        assert block is not None
        with open(rc, "w") as f:
            f.write(block)
        before = os.stat(rc).st_mtime_ns

        home = os.environ.get("HOME")
        os.environ["HOME"] = td
        try:
            ensure_babash_block_in_rc_file("/bin/bash", _Console())
        finally:
            if home is not None:
                os.environ["HOME"] = home

        assert os.stat(rc).st_mtime_ns == before, "an up-to-date rc file must not be rewritten"
