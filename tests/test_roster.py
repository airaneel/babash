"""What a reply says about sessions other than the one it is about.

An agent waiting on a slow build polls check_status every few seconds. Each of
those replies used to carry the full roster — the polled session's whole command
restated, plus every idle shell — so the same several hundred tokens came back
every time, none of them an answer to what was asked. These pin the fix: the
footer speaks only when another session has news.
"""

from dataclasses import dataclass

from babash.client.mcp_server.chat import abbreviate, full_roster, roster_footer


@dataclass
class FakeShell:
    """Just the surface roster_footer reads. A real BashState spawns a pty, which
    costs seconds; nothing here is about ptys."""

    state: str
    last_command: str
    cwd: str = "/repo"
    prompt: str | None = None

    def pending_prompt(self) -> str | None:
        return self.prompt

    def get_pending_for(self) -> str:
        return "46 seconds"


@dataclass
class FakeChat:
    chat_id: str
    main: FakeShell
    sessions: dict[str, FakeShell]


# The shape that provoked this: a long pipeline whose text dwarfs the status it
# is being polled for.
LONG_COMMAND = (
    "cd /srv/deploy && ./run-playbook.sh playbooks/site.yml -l web01 "
    "-t packages,certs 2>&1 | grep -E 'changed:|failed=|RECAP' | tail -8"
)


def busy_chat() -> FakeChat:
    return FakeChat(
        chat_id="abc123",
        main=FakeShell(state="pending", last_command=LONG_COMMAND),
        sessions={"watch": FakeShell(state="repl", last_command="ls")},
    )


def test_polling_a_session_says_nothing_about_idle_ones() -> None:
    """The reported symptom: every poll re-listed 'watch: idle', which had not
    changed and was not asked about."""
    assert roster_footer(busy_chat(), exclude="main") == ""  # type: ignore[arg-type]


def test_the_footer_does_not_restate_the_session_being_polled() -> None:
    """The status line already says what main is doing. Saying it twice, with the
    command in full both times, is what made a poll expensive."""
    footer = roster_footer(busy_chat(), exclude="main")  # type: ignore[arg-type]
    assert LONG_COMMAND not in footer


def test_another_busy_session_is_still_reported() -> None:
    """The footer exists for exactly this: news the reply itself cannot carry."""
    chat = busy_chat()
    chat.sessions["build"] = FakeShell(state="pending", last_command="npm run build")
    footer = roster_footer(chat, exclude="main")  # type: ignore[arg-type]
    assert "build: running 'npm run build' for 46 seconds" in footer
    assert "watch" not in footer  # idle, so not news


def test_another_session_blocked_on_a_question_is_still_reported() -> None:
    chat = busy_chat()
    chat.sessions["ask"] = FakeShell(
        state="pending", last_command="read -p 'ok?'", prompt="Continue? [y/N]"
    )
    footer = roster_footer(chat, exclude="main")  # type: ignore[arg-type]
    assert "WAITING FOR INPUT" in footer
    assert "send_input(text=..., session='ask')" in footer


def test_a_long_command_is_abbreviated_in_the_footer() -> None:
    chat = busy_chat()
    chat.sessions["build"] = FakeShell(state="pending", last_command=LONG_COMMAND)
    footer = roster_footer(chat, exclude="main")  # type: ignore[arg-type]
    assert LONG_COMMAND not in footer
    assert abbreviate(LONG_COMMAND) in footer
    assert len(abbreviate(LONG_COMMAND)) == 60


def test_list_sessions_still_shows_everything() -> None:
    """Trimming the incidental footer must not cost the agent the ability to ask.
    full_roster is what list_sessions returns, and it hides nothing."""
    roster = full_roster(busy_chat())  # type: ignore[arg-type]
    assert "main: running" in roster
    assert "watch: idle" in roster
    assert "[chat abc123]" in roster
