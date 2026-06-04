"""Tests for the shared RDKit physicochemical descriptor registry."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from threedscriptors.data_handling import physchem as P


def _embed(smiles: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    pos = np.asarray(conf.GetPositions())
    return nums, pos


def test_2d_descriptors_known_values():
    mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
    assert P.DESCRIPTORS_2D["n_heavy_atoms"](mol) == 6.0
    assert P.DESCRIPTORS_2D["n_aromatic_rings"](mol) == 1.0
    assert P.DESCRIPTORS_2D["mw"](mol) == pytest.approx(78.11, abs=0.1)
    assert P.DESCRIPTORS_2D["hbd"](mol) == 0.0


def test_split_names_partition_and_unknown():
    names_2d, names_3d = P.split_names(["mw", "sasa", "logp", "polar_sasa_fraction"])
    assert names_2d == ["mw", "logp"]
    assert names_3d == ["sasa", "polar_sasa_fraction"]
    with pytest.raises(KeyError):
        P.split_names(["mw", "not_a_descriptor"])


def test_3d_sasa_from_geometry():
    nums, pos = _embed("CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin
    mol3d = P.mol_from_atoms(nums, pos)
    assert mol3d is not None
    vals, ok = P.compute_3d(mol3d, ["sasa", "polar_sasa_fraction"])
    assert ok
    assert vals[0] > 0.0  # positive total SASA
    assert 0.0 <= vals[1] <= 1.0  # polar fraction is a fraction
    assert vals[1] > 0.0  # aspirin has polar O atoms


def test_polar_fraction_handles_heteroatoms():
    # Glycerol (all O) should be far more polar than benzene (no heteroatoms).
    nums_g, pos_g = _embed("OCC(O)CO")
    nums_b, pos_b = _embed("c1ccccc1")
    frac_g = P.compute_3d(P.mol_from_atoms(nums_g, pos_g), ["polar_sasa_fraction"])[0][0]
    frac_b = P.compute_3d(P.mol_from_atoms(nums_b, pos_b), ["polar_sasa_fraction"])[0][0]
    assert frac_g > frac_b
    assert frac_b == pytest.approx(0.0, abs=1e-6)


def test_compute_row_masks_missing_smiles():
    names = list(P.DEFAULT_DESCRIPTORS)
    nums, pos = _embed("CCO")
    # Valid SMILES -> everything finite/masked-in.
    vals, mask = P.compute_row(names, "CCO", nums, pos)
    assert mask.all()
    assert np.all(np.isfinite(vals))
    # Missing SMILES -> 2D masked out, 3D still computed.
    _, mask2 = P.compute_row(names, None, nums, pos)
    _, names_3d = P.split_names(names)
    for i, n in enumerate(names):
        if n in names_3d:
            assert mask2[i] == 1
        else:
            assert mask2[i] == 0


def test_compute_3d_none_mol_is_masked():
    # The explicit None contract: a failed geometry mol yields a masked NaN row.
    vals, ok = P.compute_3d(None, ["sasa", "polar_sasa_fraction"])
    assert not ok
    assert np.all(np.isnan(vals))


def test_mol_from_atoms_invalid_element_returns_none():
    # An out-of-range atomic number must be handled gracefully (no raise).
    assert P.mol_from_atoms([200], np.zeros((1, 3))) is None
