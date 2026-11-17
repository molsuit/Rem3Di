"""Pre-commit entry point: regenerate the pydantic/enum whitelist, then run vulture.

Vulture reads its paths, both whitelists and its options from ``[tool.vulture]``
in the workspace ``pyproject.toml``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = TOOLS_DIR.parents[1]


def main() -> int:
    subprocess.run(
        [sys.executable, str(TOOLS_DIR / "generate_whitelist.py")], check=True
    )
    return subprocess.run(["vulture"], cwd=WORKSPACE_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
