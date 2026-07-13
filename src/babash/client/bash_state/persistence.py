import json
import os
from typing import Any, Optional
from uuid import uuid4


def get_bash_state_dir_xdg() -> str:
    """Get the XDG directory for storing bash state."""
    xdg_data_dir = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    bash_state_dir = os.path.join(xdg_data_dir, "babash", "bash_state")
    os.makedirs(bash_state_dir, exist_ok=True)
    return bash_state_dir


def generate_thread_id() -> str:
    """Generate a collision-free thread_id.

    Used as the on-disk state-file key ({thread_id}_bash_state.json) and screen
    identity for every BashState. A 4-digit random id (the old scheme) collides
    under concurrency — two shells started at once could share a state file and
    clobber each other. A uuid removes that: distinct shells are always distinct
    on disk, which is what keeps per-chat isolation from leaking through the
    filesystem.
    """
    return f"i{uuid4().hex[:12]}"


def save_bash_state_by_id(thread_id: str, bash_state_dict: dict[str, Any]) -> None:
    """Save bash state to XDG directory with the given thread_id."""
    if not thread_id:
        return

    bash_state_dir = get_bash_state_dir_xdg()
    state_file = os.path.join(bash_state_dir, f"{thread_id}_bash_state.json")

    with open(state_file, "w") as f:
        json.dump(bash_state_dict, f, indent=2)


def load_bash_state_by_id(thread_id: str) -> Optional[dict[str, Any]]:
    """Load bash state from XDG directory with the given thread_id."""
    if not thread_id:
        return None

    bash_state_dir = get_bash_state_dir_xdg()
    state_file = os.path.join(bash_state_dir, f"{thread_id}_bash_state.json")

    if not os.path.exists(state_file):
        return None

    with open(state_file) as f:
        return json.load(f)  # type: ignore
