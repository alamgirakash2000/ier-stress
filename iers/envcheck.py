"""Startup guard: every script must run inside the `ierstress` conda environment."""

import sys
from pathlib import Path

REQUIRED_ENV = "ierstress"


def current_env_name() -> str:
    return Path(sys.prefix).name


def require_ierstress() -> None:
    name = current_env_name()
    if name != REQUIRED_ENV:
        sys.exit(
            f"ERROR: this script must run inside the '{REQUIRED_ENV}' conda environment, "
            f"but it is running in '{name}' ({sys.prefix}).\n"
            f"Run: conda activate {REQUIRED_ENV}"
        )
