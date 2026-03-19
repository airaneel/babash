# babash

Shell and coding agent MCP server. Fork of [wcgw](https://github.com/rusiaaman/wcgw), modernized with FastMCP and decomposed architecture.

## Features

- **Interactive shell** — fully interactive terminal via pexpect/pyte, supporting arrow keys, Ctrl-C, background processes
- **Smart file editing** — search/replace with fuzzy matching, indentation tolerance, syntax checking on writes
- **File protections** — read-before-edit enforcement, hash-based change detection, token-aware truncation with temp file save
- **Three modes** — `wcgw` (full access), `architect` (read-only), `code_writer` (restricted paths/commands)
- **Task persistence** — save/resume context across chat sessions via ContextSave
- **ML file ranking** — pre-trained model scores file importance for smart repo overviews
- **Streamable HTTP** — supports both stdio and streamable-http transports

## Setup

### Claude Desktop

Install `uv`, then add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "babash": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/airaneel/babash", "babash"]
    }
  }
}
```

To force a specific shell:

```json
{
  "mcpServers": {
    "babash": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/airaneel/babash", "babash", "--shell", "/bin/zsh"]
    }
  }
}
```

### Streamable HTTP (remote deployment)

```bash
babash_mcp --transport streamable-http
```

### Docker

```bash
docker build -t babash https://github.com/airaneel/babash.git
docker run -i --rm --mount type=bind,src=/your/workspace,dst=/workspace babash
```

## Tools

| Tool | Description |
|---|---|
| `Initialize` | Set up workspace, mode, resume tasks |
| `BashCommand` | Execute shell commands, check status, send keystrokes |
| `ReadFiles` | Read files with optional line ranges (`file.py:10-20`) |
| `ReadImage` | Read image files |
| `FileWriteOrEdit` | Write new files or edit existing ones (auto-selects mode by % changed) |
| `ContextSave` | Save task context + relevant files for later resumption |

## Modes

| Mode | Shell | File Edit | File Write |
|---|---|---|---|
| `wcgw` (default) | Full access | All files | All files |
| `architect` | Read-only | None | None |
| `code_writer` | Configurable | Specified globs | Specified globs |

## Terminal attachment

If `screen` is installed, babash runs in a screen session. Attach with:

```bash
screen -ls          # find the session
screen -x <id>      # attach (use Ctrl+A+D to detach)
```

## Architecture

```
src/babash/
├── __init__.py                     # Entry point
├── types_.py                       # Pydantic models for all tool inputs
└── client/
    ├── mcp_server/
    │   ├── __init__.py             # Typer CLI (stdio + streamable-http)
    │   └── server.py              # FastMCP server with lifespan
    ├── bash_state/
    │   ├── bash_state.py          # Core BashState class
    │   ├── shell_process.py       # pexpect/pyte, screen sessions
    │   ├── execute.py             # Command execution engine
    │   ├── file_whitelist.py      # Read-before-edit tracking
    │   └── persistence.py         # State serialization to disk
    ├── file_ops/
    │   ├── search_replace.py      # Aider-style search/replace
    │   ├── diff_edit.py           # Fuzzy matching engine
    │   └── extensions.py          # Language detection, token limits
    ├── repo_ops/                  # Git-aware repo analysis, ML ranking
    ├── tools.py                   # Tool dispatch and file operations
    ├── modes.py                   # Mode definitions and prompts
    ├── tool_prompts.py            # MCP tool schemas and descriptions
    └── encoder/                   # Lazy-loaded tokenizer
```

## Development

```bash
uv sync
uv run mypy --strict src/babash    # type checking
uv run pytest                       # tests
```

## Credits

Fork of [rusiaaman/wcgw](https://github.com/rusiaaman/wcgw). Modernized with FastMCP, decomposed BashState, cleaned up types.
