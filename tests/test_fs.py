"""The file layer, exercised identically against both stores.

Every test that can run against a local file also runs against one reached
through a live shell — that's what `store` being parametrized buys. It's the
whole point of the FileStore seam: if the two ever diverge, a test fails.
"""

import os
from typing import Iterator

import pytest

from babash.client.bash_state import BashState
from babash.client.fs import FileError, FileStore, LocalStore, SessionStore

# Content that a heredoc would mangle and base64 does not: quotes, $, backticks,
# and a line that looks like a heredoc terminator.
NASTY = """key: "a $VALUE with `backticks` and 'quotes'"
script: |
  echo "don't ${EXPAND} me" > /dev/null
  printf '%s\\n' "$(whoami)"
EOF
tail: done
"""


@pytest.fixture(params=["local", "session"])
def store(request: pytest.FixtureRequest, shell: BashState) -> Iterator[FileStore]:
    if request.param == "local":
        yield LocalStore()
    else:
        yield SessionStore(shell, "test-session")


def test_write_then_read_roundtrips(store: FileStore, temp_dir: str) -> None:
    path = os.path.join(temp_dir, "hello.txt")
    store.write(path, "hello\nworld\n")
    assert store.read(path) == "hello\nworld\n"


def test_shell_metacharacters_survive(store: FileStore, temp_dir: str) -> None:
    """The reason write_file exists instead of `cat <<EOF`."""
    path = os.path.join(temp_dir, "config.yaml")
    store.write(path, NASTY)
    assert store.read(path) == NASTY


def test_write_creates_parent_directories(store: FileStore, temp_dir: str) -> None:
    path = os.path.join(temp_dir, "deep", "deeper", "file.txt")
    store.write(path, "x")
    assert store.exists(path)


def test_write_overwrites(store: FileStore, temp_dir: str) -> None:
    path = os.path.join(temp_dir, "twice.txt")
    store.write(path, "first")
    store.write(path, "second")
    assert store.read(path) == "second"


def test_exists(store: FileStore, temp_dir: str) -> None:
    path = os.path.join(temp_dir, "here.txt")
    assert not store.exists(path)
    store.write(path, "x")
    assert store.exists(path)


def test_reading_a_missing_file_says_so(store: FileStore, temp_dir: str) -> None:
    with pytest.raises(FileError, match="does not exist"):
        store.read(os.path.join(temp_dir, "nope.txt"))


def test_empty_file_roundtrips(store: FileStore, temp_dir: str) -> None:
    path = os.path.join(temp_dir, "empty.txt")
    store.write(path, "")
    assert store.read(path) == ""


def test_unicode_survives(store: FileStore, temp_dir: str) -> None:
    path = os.path.join(temp_dir, "unicode.txt")
    content = "привет — ◉ ➤ 日本語\n"
    store.write(path, content)
    assert store.read(path) == content


def test_session_store_refuses_while_the_shell_is_busy(
    shell: BashState, temp_dir: str
) -> None:
    """A command is still running, so anything we sent would go to *it*, not bash."""
    from babash.client.bash_state import execute_bash
    from babash.types_ import Command

    execute_bash(shell, Command(command="sleep 30"), None, None)
    store = SessionStore(shell, "busy")
    with pytest.raises(FileError, match="busy"):
        store.read(os.path.join(temp_dir, "anything.txt"))


def test_binary_survives_the_pty(store: FileStore, temp_dir: str) -> None:
    """An image has to cross a terminal to get here, and a terminal will happily
    read some of those bytes as control sequences. base64 is why it doesn't."""
    path = os.path.join(temp_dir, "image.png")
    # Every byte value, including the ones a tty would otherwise act on.
    blob = bytes(range(256)) * 8
    with open(path, "wb") as f:
        f.write(blob)

    assert store.read_bytes(path) == blob


def test_read_bytes_on_a_missing_file(store: FileStore, temp_dir: str) -> None:
    with pytest.raises(FileError, match="does not exist"):
        store.read_bytes(os.path.join(temp_dir, "nope.png"))
