import json
import os
import random
from typing import Any, Optional


def get_bash_state_dir_xdg() -> str:
    """Get the XDG directory for storing bash state."""
    xdg_data_dir = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    bash_state_dir = os.path.join(xdg_data_dir, "wcgw", "bash_state")
    os.makedirs(bash_state_dir, exist_ok=True)
    return bash_state_dir


def generate_thread_id() -> str:
    """Generate a random 4-digit thread_id."""
    return f"i{random.randint(1000, 9999)}"


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
