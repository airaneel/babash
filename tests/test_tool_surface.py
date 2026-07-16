"""The tool surface is prompt: schemas, descriptions and annotations ship to the
model on every session, and nothing here is checked by mypy or ruff.

These assert against the registered tools rather than the source, so they see
what the model sees.
"""

import asyncio

import pytest
from mcp.types import Tool

from babash.client.mcp_server import server  # noqa: F401  — registers every tool
from babash.client.mcp_server.instance import mcp
from babash.client.mcp_server.tools.files import _numbered


@pytest.fixture(scope="module")
def tools() -> dict[str, Tool]:
    return {t.name: t for t in asyncio.run(mcp.list_tools())}


def test_every_tool_is_annotated(tools: dict[str, Tool]) -> None:
    missing = [n for n, t in tools.items() if t.annotations is None]
    assert not missing, f"tools without annotations: {missing}"


def test_readers_do_not_claim_destructive_or_idempotent(tools: dict[str, Tool]) -> None:
    """destructiveHint and idempotentHint are only meaningful when readOnlyHint is
    False. Setting them on a reader is noise at best; on check_status
    idempotentHint=True was also a lie, since it returns output *since the last
    check* and advances that cursor."""
    for name, t in tools.items():
        a = t.annotations
        assert a is not None
        if a.readOnlyHint:
            assert a.destructiveHint is None, f"{name} sets destructiveHint but is read-only"
            assert a.idempotentHint is None, f"{name} sets idempotentHint but is read-only"


def test_tools_that_can_destroy_say_so(tools: dict[str, Tool]) -> None:
    """A client uses destructiveHint to decide whether to confirm with the user.
    babash_initialize re-run on a known chat_id kills that chat's shells, taking
    any running command with them — it is not an additive update."""
    for name in ("run_command", "write_file", "edit_file", "destroy_session", "babash_initialize"):
        a = tools[name].annotations
        assert a is not None
        assert a.destructiveHint is True, f"{name} can destroy state but does not say so"


def test_chat_id_is_described_wherever_it_is_taken(tools: dict[str, Tool]) -> None:
    """chat_id is the one parameter the whole isolation model rests on. An
    unannotated `str` compiles to {"title": "Chat Id", "type": "string"}, which
    tells the model nothing about where the value comes from."""
    for name, t in tools.items():
        prop = t.inputSchema.get("properties", {}).get("chat_id")
        if prop is not None:
            assert prop.get("description"), f"{name}.chat_id has no description"


def test_read_file_bounds_are_in_the_schema(tools: dict[str, Tool]) -> None:
    """Constraints are how the model self-corrects: a rejected offset=0 teaches
    it, where the old silent max(1, offset) clamp did not."""
    props = tools["read_file"].inputSchema["properties"]
    assert props["offset"]["minimum"] == 1
    assert props["limit"]["minimum"] == 1
    assert props["offset"]["description"]
    assert props["limit"]["description"]


def test_a_read_past_the_end_does_not_claim_the_file_is_empty() -> None:
    """The old answer to both was "(empty file)" — so reading a perfectly good
    file at a stale offset told the model it had nothing in it."""
    out = _numbered("line1\nline2\nline3", offset=99999, limit=10)
    assert "empty" not in out
    assert "3 lines" in out


def test_an_actually_empty_file_still_says_so() -> None:
    assert _numbered("", offset=1, limit=10) == "(empty file)"


def test_numbering_starts_at_the_requested_offset() -> None:
    out = _numbered("a\nb\nc", offset=2, limit=1)
    assert out.startswith("     2\tb")
    assert "pass offset=3 to continue" in out
