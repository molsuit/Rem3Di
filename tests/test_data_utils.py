import numpy as np
import pytest

from remedi.data_handling.data_utils import (
    get_max_num_of_heavy_atom_from_atoms,
)


class DummyMol:
    """Minimal stand-in for an ASE Atoms object."""

    def __init__(self, atomic_numbers):
        self._numbers = np.array(atomic_numbers)

    def get_atomic_numbers(self):
        return self._numbers


@pytest.mark.parametrize(
    "atom_seqs, expected",
    [
        ([], 0),  # no molecules at all
        ([[1, 1, 1, 1]], 0),  # only hydrogens → 0 heavies
        ([[1, 6, 1, 8, 1]], 2),  # single mixed molecule
        ([[1, 1, 7, 16, 16]], 3),  # another single example
        (
            [
                [1, 6, 6, 8],
                [1, 1, 7, 16, 16],  # multiple molecules, pick max
                [6, 6, 6, 6, 6],
            ],
            5,
        ),
    ],
    ids=[
        "no-mols",
        "all-H",
        "one-mixed",
        "one-varied-heavy",
        "multiple-mols",
    ],
)
def test_get_max_heavy(atom_seqs, expected):
    """get_max always returns the highest count of non-H atoms across molecules."""
    mols = [DummyMol(seq) for seq in atom_seqs]
    assert get_max_num_of_heavy_atom_from_atoms(mols) == expected
