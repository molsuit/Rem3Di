"""Clash detection, standardization and the stereochemistry gates."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import ALANINE, coordinates, embed
from rdkit import Chem

from chiralcat_dataset.chemistry import (
    Clash,
    add_hydrogens,
    canonical_smiles,
    categorize_clash,
    chirality_preserved,
    count_stereocenters_from_3d,
    find_clashes,
    has_unspecified_stereo,
    identity_signature,
    standardize_mol,
    stereo_signature_from_3d,
)


@pytest.mark.parametrize(
    ("symbols", "distance", "expected_clashes"),
    [
        (["C", "C"], 0.5, 1),
        (["C", "C"], 1.5, 0),  # a normal bond
        (["C", "C"], 0.72, 1),
        (["H", "H"], 0.72, 0),  # the floor is element-aware
        (["C", "H"], 0.72, 1),
    ],
)
def test_find_clashes_uses_an_element_aware_floor(symbols, distance, expected_clashes):
    coords = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    assert len(find_clashes(symbols, coords)) == expected_clashes


def test_find_clashes_sorts_nearest_first_and_handles_a_single_atom():
    coords = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.3, 0.0]])
    distances = [clash.distance for clash in find_clashes(["C", "C", "C"], coords)]
    assert distances == sorted(distances) and len(distances) == 3
    assert find_clashes(["C"], np.array([[0.0, 0.0, 0.0]])) == []


@pytest.mark.parametrize(
    ("symbols", "smiles", "distance", "category"),
    [
        (["C", "C"], "CCO", 0.5, "collapsed_geometry_organic"),
        (["C", "H"], "CCO", 0.5, "hydrogen_artifact"),
        (["C", "Cl"], "C[C@H](N)C(=O)O.[Cl-]", 0.5, "multi_fragment_salt"),
        (["Fe", "C"], "[Fe].c1ccccc1", 0.5, "metal_organometallic"),
        (["Fe", "C"], "C[Fe]", 0.5, "metal_organometallic"),
        (["C", "C"], "CCO", 0.05, "exact_dup_organometallic"),  # wins over all
    ],
)
def test_categorize_clash(symbols, smiles, distance, category):
    clash = Clash(0, symbols[0], 1, symbols[1], distance)
    assert categorize_clash(symbols, smiles, [clash]) == category


def test_standardize_mol_is_stereo_safe():
    mol = embed(ALANINE + ".[Cl-]")
    before = Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)

    standardized = standardize_mol(mol, strip_salts=True, neutralize=True)
    assert standardized is not None
    after = Chem.FindMolChiralCenters(standardized, useLegacyImplementation=False)

    assert before == after  # same atom index, same R/S
    assert "." not in Chem.MolToSmiles(standardized)  # counter-ion removed
    assert all(atom.GetFormalCharge() == 0 for atom in standardized.GetAtoms())


def test_standardize_mol_keeps_a_bare_salt():
    # dontRemoveEverything keeps the last fragment rather than emptying the mol.
    mol = Chem.MolFromSmiles("[Cl-]")
    assert standardize_mol(mol, strip_salts=True, neutralize=False) is not None


def test_canonical_smiles_returns_none_for_garbage():
    assert canonical_smiles("not a molecule") is None
    assert canonical_smiles("CCO") == Chem.CanonSmiles("CCO")


def test_add_hydrogens_places_every_hydrogen_within_bonding_distance():
    with_hydrogens = add_hydrogens(Chem.RemoveHs(embed("CC(C)(C)[C@H](O)c1ccccc1")))
    coords = coordinates(with_hydrogens)
    is_hydrogen = np.array(
        [atom.GetSymbol() == "H" for atom in with_hydrogens.GetAtoms()]
    )
    heavy = coords[~is_hydrogen]
    for hydrogen in coords[is_hydrogen]:
        assert float(np.min(np.linalg.norm(heavy - hydrogen, axis=1))) <= 1.4


@pytest.mark.parametrize(
    ("smiles", "unspecified"),
    [("CC(N)C(=O)O", True), (ALANINE, False), ("CCO", False)],
)
def test_has_unspecified_stereo(smiles, unspecified):
    assert has_unspecified_stereo(smiles) is unspecified


@pytest.mark.parametrize(
    ("intended", "preserved"), [(ALANINE, True), ("C[C@@H](N)C(=O)O", False)]
)
def test_chirality_gate_compares_the_geometry_with_the_intended_enantiomer(
    intended, preserved
):
    assert chirality_preserved(intended, embed(ALANINE))[0] is preserved


def test_signatures_read_the_geometry_and_drop_hydrogens():
    mol = embed(ALANINE)
    assert stereo_signature_from_3d(mol) == Chem.CanonSmiles(ALANINE)
    assert identity_signature(mol) == Chem.CanonSmiles(ALANINE)


@pytest.mark.parametrize(("smiles", "assigned"), [(ALANINE, 1), ("CCO", 0)])
def test_count_stereocenters_from_3d(smiles, assigned):
    found, tagged = count_stereocenters_from_3d(embed(smiles))
    assert found == assigned
    assert tagged >= assigned and (tagged == 0) == (assigned == 0)
