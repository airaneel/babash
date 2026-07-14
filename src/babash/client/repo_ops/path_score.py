"""Heuristic ranking of file paths by likely importance to a reader.

Replaces a shipped unigram language model (a 1.9 MB tokenizer + vocab, pulling
in the `tokenizers` dependency). That model summed per-token log-probabilities,
which in practice made the score ~linear in token count — i.e. it ranked
"short paths made of common path fragments" first, and nothing more. It also
carried a JS/Java-skewed vocabulary from upstream.

The signals that actually carry information — git recency and the workspace
read/edit counts — are applied by the caller *before* this, so this function
only orders the tail of the file list. A few explicit rules do that at least as
well as the model did.
"""

# Files that are almost always worth surfacing when they sit at the repo root.
_IMPORTANT_ROOT_NAMES = frozenset(
    {
        "readme.md",
        "readme.rst",
        "readme.txt",
        "readme",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "package.json",
        "tsconfig.json",
        "go.mod",
        "cargo.toml",
        "pom.xml",
        "build.gradle",
        "makefile",
        "dockerfile",
        "docker-compose.yml",
        "claude.md",
        "agents.md",
    }
)

# Directories whose contents are rarely what a reader is looking for.
_NOISE_DIRS = frozenset(
    {
        "node_modules",
        "venv",
        ".venv",
        "dist",
        "build",
        "target",
        "vendor",
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "site-packages",
        "htmlcov",
        ".tox",
        ".idea",
        ".vscode",
        "coverage",
        "migrations",
    }
)

# Top-level directories that usually hold the code worth reading.
_SOURCE_DIRS = frozenset({"src", "lib", "app", "cmd", "internal", "pkg", "tests", "test"})

# Extensions that are generated or otherwise low-signal.
_LOW_SIGNAL_SUFFIXES = (".min.js", ".min.css", ".lock", ".map", ".snap", ".pyc")


def score_path(path: str) -> float:
    """Importance of a repo-relative path. Higher is more important."""
    parts = path.split("/")
    name = parts[-1].lower()
    dirs = [p.lower() for p in parts[:-1]]

    # Shallow files are usually the entry points; each level of nesting costs.
    score = -float(len(parts))

    if any(d in _NOISE_DIRS for d in dirs):
        score -= 100.0
    if len(parts) == 1 and name in _IMPORTANT_ROOT_NAMES:
        score += 50.0
    if dirs and dirs[0] in _SOURCE_DIRS:
        score += 10.0
    if name.startswith("."):
        score -= 5.0
    if name.endswith(_LOW_SIGNAL_SUFFIXES):
        score -= 20.0

    return score


def sort_paths_by_importance(paths: list[str]) -> list[str]:
    """Most important first. Ties keep their original (stable) order."""
    return sorted(paths, key=score_path, reverse=True)
