"""Repair strategies and the gates that refuse an unsafe repair."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from chiralcat_dataset.chemistry import (
    chirality_preserved,
    find_clashes,
    identity_signature,
    stereo_signature_from_3d,
)
from chiralcat_dataset.records import BROKEN, FILTERED, Structure
from chiralcat_dataset.repair import reembed, repair, repair_structure, strip_counterion

SEED = 0xC0FFEE


def _coords(mol: Chem.Mol) -> np.ndarray:
    conformer = mol.GetConformer()
    return np.array(
        [list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )


# --------------------------------------------------------------------------- #
# Re-embedding
# --------------------------------------------------------------------------- #


def test_reembed_resolves_clash_and_preserves_chirality():
    smiles = "CC(C)(C)[C@H](O)c1ccccc1"  # the collapsed tert-butyl case
    mol = reembed(smiles, seed=SEED, mmff_iters=2000)
    assert mol is not None
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert find_clashes(symbols, _coords(mol)) == []
    preserved, _ = chirality_preserved(smiles, mol)
    assert preserved is True


def test_reembed_returns_none_for_unparseable_smiles():
    assert reembed("not a molecule", seed=SEED, mmff_iters=10) is None


# --------------------------------------------------------------------------- #
# Counter-ion stripping (original coordinates preserved)
# --------------------------------------------------------------------------- #


def _heavy_only_salt(smiles: str) -> Chem.Mol:
    """Embed a salt and strip Hs -> a heavy-atom mol with a conformer."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    AllChem.EmbedMolecule(mol, params)
    return Chem.RemoveHs(mol)


def test_strip_counterion_keeps_largest_fragment():
    repaired = strip_counterion(_heavy_only_salt("C[C@H](N)C(=O)O.[Cl-]"), 0)
    assert repaired is not None
    assert identity_signature(repaired) == Chem.CanonSmiles("C[C@H](N)C(=O)O")


def test_strip_counterion_preserves_3d_chirality():
    repaired = strip_counterion(_heavy_only_salt("C[C@H](N)C(=O)O.[Cl-]"), 0)
    assert repaired is not None
    assert stereo_signature_from_3d(repaired) == Chem.CanonSmiles("C[C@H](N)C(=O)O")


def test_strip_counterion_returns_none_for_single_fragment():
    assert strip_counterion(_heavy_only_salt("C[C@H](N)C(=O)O"), 0) is None


# --------------------------------------------------------------------------- #
# Stage behaviour
# --------------------------------------------------------------------------- #


def _structure(index: int, smiles: str, class_name: str, symbols, coords) -> Structure:
    from chiralcat_dataset.taxonomy import CLASS_TO_LABEL

    return Structure(
        index=index,
        smiles=smiles,
        class_name=class_name,
        label=CLASS_TO_LABEL[class_name],
        symbols=list(symbols),
        coords=[tuple(float(v) for v in point) for point in coords],
        source_file="test.pkl",
    )


def test_clash_free_structure_passes_through_untouched(config):
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))
    AllChem.EmbedMolecule(mol, randomSeed=SEED)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    structure = _structure(0, "C[C@H](N)C(=O)O", "central", symbols, _coords(mol))

    output = repair(config, [structure], {})
    assert len(output.structures) == 1
    assert output.structures[0].geometry_quality == "ok"
    assert output.rejected == []


def test_unrepairable_structure_is_rejected_as_broken_with_geometry(config):
    # A metal organometallic is out of repair scope: flagged, never repaired.
    structure = _structure(
        0, "[Fe].c1ccccc1", "planar", ["Fe", "C"], [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]
    )
    output = repair(config, [structure], {})
    assert output.structures == []
    assert len(output.rejected) == 1
    rejected = output.rejected[0]
    assert rejected.disposition == BROKEN
    assert rejected.stage == "repair"
    assert rejected.has_geometry is True  # geometry survives for inspection


def test_reembeddable_clash_is_repaired_and_marked(config):
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))
    AllChem.EmbedMolecule(mol, randomSeed=SEED)
    coords = _coords(mol)
    coords[1] = coords[0] + np.array([0.3, 0.0, 0.0])  # collapse two atoms
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    structure = _structure(0, "C[C@H](N)C(=O)O", "central", symbols, coords)

    output = repair(config, [structure], {})
    assert len(output.structures) == 1
    repaired = output.structures[0]
    assert repaired.geometry_quality == "repaired"
    assert repaired.repair_strategy == "reembed"
    assert find_clashes(repaired.symbols, np.array(repaired.coords)) == []


def test_axial_clash_is_not_reembedded_because_smiles_cannot_encode_it(config):
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)(C)C(O)c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=SEED)
    coords = _coords(mol)
    coords[1] = coords[0] + np.array([0.3, 0.0, 0.0])
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    structure = _structure(0, "CC(C)(C)C(O)c1ccccc1", "axial", symbols, coords)

    outcome = repair_structure(structure, config, {})
    assert outcome.status == "flagged"
    assert "not SMILES-encoded" in outcome.note


def test_disabled_repair_stage_passes_everything_through(config):
    config = config.model_copy(deep=True)
    config.repair.enabled = False
    structure = _structure(
        0, "[Fe].c1ccccc1", "planar", ["Fe", "C"], [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]
    )
    output = repair(config, [structure], {})
    assert len(output.structures) == 1
    assert output.rejected == []


def test_duplicate_collapse_is_recorded_as_filtered_not_broken(config):
    """Two entries that resolve to the same SMILES keep one and file the other."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))
    AllChem.EmbedMolecule(mol, randomSeed=SEED)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = _coords(mol)
    first = _structure(0, "C[C@H](N)C(=O)O", "central", symbols, coords)
    second = _structure(1, "C[C@H](N)C(=O)O", "central", symbols, coords)

    output = repair(config, [first, second], {})
    assert len(output.structures) == 1
    assert len(output.rejected) == 1
    assert output.rejected[0].disposition == FILTERED
    assert output.rejected[0].reason == "dropped_duplicate"
