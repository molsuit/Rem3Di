"""Clash detection, standardization and the stereochemistry gates."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from chiralcat_dataset.chemistry import (
    add_hydrogens,
    canonical_smiles,
    categorize_clash,
    chirality_preserved,
    count_stereocenters_from_3d,
    find_clashes,
    has_unspecified_stereo,
    identity_signature,
    pair_floor,
    standardize_mol,
    stereo_signature_from_3d,
)

SEED = 0xC0FFEE


# --------------------------------------------------------------------------- #
# Clash detection
# --------------------------------------------------------------------------- #


def test_find_clashes_flags_close_heavy_pair():
    coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    clashes = find_clashes(["C", "C"], coords)
    assert len(clashes) == 1
    assert clashes[0].distance == 0.5


def test_find_clashes_respects_normal_bond():
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    assert find_clashes(["C", "C"], coords) == []


def test_find_clashes_element_aware_floor_for_hydrogens():
    # 0.72 A is a clash between heavy atoms but acceptable for an H-H pair.
    coords = np.array([[0.0, 0.0, 0.0], [0.72, 0.0, 0.0]])
    assert find_clashes(["H", "H"], coords) == []
    assert len(find_clashes(["C", "C"], coords)) == 1


def test_pair_floor_ordering():
    assert pair_floor("H", "H") < pair_floor("C", "H") < pair_floor("C", "C")


def test_find_clashes_sorts_nearest_first():
    coords = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.3, 0.0]])
    clashes = find_clashes(["C", "C", "C"], coords)
    distances = [clash.distance for clash in clashes]
    assert distances == sorted(distances)


def test_find_clashes_handles_single_atom():
    assert find_clashes(["C"], np.array([[0.0, 0.0, 0.0]])) == []


# --------------------------------------------------------------------------- #
# Clash categorisation
# --------------------------------------------------------------------------- #


def _clash(distance: float, symbol_a: str = "C", symbol_b: str = "C"):
    from chiralcat_dataset.chemistry import Clash

    return Clash(0, symbol_a, 1, symbol_b, distance)


def test_categorize_collapsed_organic():
    assert categorize_clash(["C", "C"], "CCO", [_clash(0.5)]) == (
        "collapsed_geometry_organic"
    )


def test_categorize_hydrogen_artifact():
    category = categorize_clash(["C", "H"], "CCO", [_clash(0.5, "C", "H")])
    assert category == "hydrogen_artifact"


def test_categorize_multi_fragment_salt():
    category = categorize_clash(["C", "Cl"], "C[C@H](N)C(=O)O.[Cl-]", [_clash(0.5)])
    assert category == "multi_fragment_salt"


def test_categorize_metal_organometallic():
    category = categorize_clash(["Fe", "C"], "[Fe].c1ccccc1", [_clash(0.5)])
    assert category == "metal_organometallic"


def test_categorize_exact_duplicate_wins_over_everything():
    category = categorize_clash(["C", "C"], "CCO", [_clash(0.05)])
    assert category == "exact_dup_organometallic"


# --------------------------------------------------------------------------- #
# Standardization
# --------------------------------------------------------------------------- #


def test_standardize_mol_is_stereo_safe():
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O.[Cl-]"))
    AllChem.EmbedMolecule(mol, randomSeed=SEED)
    before = Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)

    standardized = standardize_mol(mol, strip_salts=True, neutralize=True)
    after = Chem.FindMolChiralCenters(standardized, useLegacyImplementation=False)

    assert before == after  # same atom index, same R/S
    assert "." not in Chem.MolToSmiles(standardized)  # counter-ion removed
    assert all(atom.GetFormalCharge() == 0 for atom in standardized.GetAtoms())


def test_standardize_mol_returns_none_when_everything_is_stripped():
    # dontRemoveEverything keeps the last fragment, so a bare salt survives.
    mol = Chem.MolFromSmiles("[Cl-]")
    assert standardize_mol(mol, strip_salts=True, neutralize=False) is not None


def test_canonical_smiles_returns_none_for_garbage():
    assert canonical_smiles("not a molecule") is None
    assert canonical_smiles("CCO") == Chem.CanonSmiles("CCO")


# --------------------------------------------------------------------------- #
# Hydrogen placement
# --------------------------------------------------------------------------- #


def test_add_hydrogens_places_every_hydrogen_within_bonding_distance():
    mol = Chem.MolFromSmiles("CC(C)(C)[C@H](O)c1ccccc1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=SEED)
    heavy_only = Chem.RemoveHs(mol)

    with_hydrogens = add_hydrogens(heavy_only)
    conformer = with_hydrogens.GetConformer()
    coords = np.array(
        [list(conformer.GetAtomPosition(i)) for i in range(with_hydrogens.GetNumAtoms())]
    )
    symbols = [atom.GetSymbol() for atom in with_hydrogens.GetAtoms()]
    heavy = coords[[i for i, s in enumerate(symbols) if s != "H"]]
    for i, symbol in enumerate(symbols):
        if symbol != "H":
            continue
        assert float(np.min(np.linalg.norm(heavy - coords[i], axis=1))) <= 1.4


# --------------------------------------------------------------------------- #
# Stereochemistry
# --------------------------------------------------------------------------- #


def test_unspecified_stereo_true_for_missing_descriptor():
    assert has_unspecified_stereo("CC(N)C(=O)O") is True


def test_unspecified_stereo_false_when_specified():
    assert has_unspecified_stereo("C[C@H](N)C(=O)O") is False


def test_unspecified_stereo_false_when_no_stereocentre():
    assert has_unspecified_stereo("CCO") is False


def test_chirality_gate_accepts_matching_enantiomer():
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))
    AllChem.EmbedMolecule(mol, randomSeed=SEED)
    preserved, _ = chirality_preserved("C[C@H](N)C(=O)O", mol)
    assert preserved is True


def test_chirality_gate_rejects_enantiomer():
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))
    AllChem.EmbedMolecule(mol, randomSeed=SEED)
    preserved, _ = chirality_preserved("C[C@@H](N)C(=O)O", mol)
    assert preserved is False


def test_stereo_signature_reads_geometry_not_flags():
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))
    AllChem.EmbedMolecule(mol, randomSeed=SEED)
    assert stereo_signature_from_3d(mol) == Chem.CanonSmiles("C[C@H](N)C(=O)O")


def test_identity_signature_strips_hydrogens():
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))
    assert identity_signature(mol) == Chem.CanonSmiles("C[C@H](N)C(=O)O")


def test_count_stereocenters_from_3d_finds_the_single_centre():
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))
    AllChem.EmbedMolecule(mol, randomSeed=SEED)
    assigned, tagged = count_stereocenters_from_3d(mol)
    assert assigned == 1
    assert tagged >= 1


def test_count_stereocenters_from_3d_finds_none_for_achiral():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=SEED)
    assigned, tagged = count_stereocenters_from_3d(mol)
    assert assigned == 0
    assert tagged == 0
