"""Shared fixtures for the remedi tests.

``torch_sim_compiler`` skips tests that run MACE through torch-sim: its
neighbour list is compiled by torch inductor, which needs gcc >= 10 (the DAIS
system gcc is 7.5; ``module load gcc/15``). Request it with
``@pytest.mark.usefixtures("torch_sim_compiler")`` or as a fixture argument.
"""

from __future__ import annotations

import subprocess

import pytest


def _gcc_major_version() -> int:
    """Major version of the ``gcc`` on PATH, 0 if there is none."""
    try:
        completed = subprocess.run(
            ["gcc", "-dumpversion"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return 0
    return int(completed.stdout.strip().split(".")[0])


@pytest.fixture(scope="session")
def torch_sim_compiler() -> None:
    if _gcc_major_version() < 10:
        pytest.skip(
            "torch inductor needs gcc >= 10 to build torch-sim's neighbour list "
            "(on DAIS: module load gcc/15)"
        )
