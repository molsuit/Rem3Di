"""The 2D RDKit descriptor registry the dataset analysis reads."""

from __future__ import annotations

import pytest
from rdkit import Chem

from remedi_eval.dataset_analysis.physchem import DESCRIPTORS_2D


def test_2d_descriptors_known_values():
    mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
    assert DESCRIPTORS_2D["n_heavy_atoms"](mol) == 6.0
    assert DESCRIPTORS_2D["n_aromatic_rings"](mol) == 1.0
    assert DESCRIPTORS_2D["mw"](mol) == pytest.approx(78.11, abs=0.1)
    assert DESCRIPTORS_2D["hbd"](mol) == 0.0
