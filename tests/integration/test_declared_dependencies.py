"""Every workspace member declares exactly the third-party packages it imports.

deptry runs on each member's own directory with the member's ``pyproject.toml``
(its ``[tool.deptry]`` table lists the deliberate exceptions, each with its
reason): an imported but undeclared package (DEP001, DEP003) or a declared but
unused one (DEP002) fails. Extras count as declared, so a module that needs an
extra imports it where it is used and names the extra when it is missing.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MEMBERS = [
    "packages/remedi-data",
    "packages/remedi-eval",
    "packages/remedi",
    "preparers/tdc",
    "preparers/moleculenet",
    "preparers/polaris",
    "preparers/chiralcat",
    "preparers/qm9or",
]


def test_the_member_list_is_the_workspace() -> None:
    import tomllib

    workspace = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]
    assert sorted(workspace["workspace"]["members"]) == sorted(MEMBERS)


@pytest.mark.parametrize("member", MEMBERS)
def test_declared_dependencies_match_the_imports(member: str) -> None:
    deptry = shutil.which("deptry", path=str(Path(sys.executable).parent))
    assert deptry is not None, "deptry is in the root dev dependency group"
    completed = subprocess.run(
        [deptry, "."],
        cwd=ROOT / member,
        capture_output=True,
        text=True,
        env={"NO_COLOR": "1", "PATH": str(Path(sys.executable).parent)},
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
