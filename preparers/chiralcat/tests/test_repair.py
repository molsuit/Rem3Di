"""Repair strategies and the gates that refuse an unsafe repair."""

from __future__ import annotations

import numpy as np
from conftest import ALANINE, SEED, coordinates, embed, make_structure
from rdkit import Chem

from chiralcat_dataset.chemistry import (
    chirality_preserved,
    find_clashes,
    identity_signature,
    stereo_signature_from_3d,
)
from chiralcat_dataset.records import BROKEN, FILTERED, Structure
from chiralcat_dataset.repair import reembed, repair, repair_structure, strip_counterion


def test_reembed_resolves_clash_and_preserves_chirality():
    smiles = "CC(C)(C)[C@H](O)c1ccccc1"  # the collapsed tert-butyl case
    mol = reembed(smiles, seed=SEED, mmff_iters=2000)
    assert mol is not None
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert find_clashes(symbols, coordinates(mol)) == []
    assert chirality_preserved(smiles, mol)[0] is True
    assert reembed("not a molecule", seed=SEED, mmff_iters=10) is None


def test_strip_counterion_keeps_the_cation_and_its_3d_chirality():
    salt = Chem.RemoveHs(embed(ALANINE + ".[Cl-]"))
    repaired = strip_counterion(salt, 0)
    assert repaired is not None
    assert identity_signature(repaired) == Chem.CanonSmiles(ALANINE)
    assert stereo_signature_from_3d(repaired) == Chem.CanonSmiles(ALANINE)
    assert strip_counterion(Chem.RemoveHs(embed(ALANINE)), 0) is None


def _collapsed(smiles: str, class_name: str) -> Structure:
    structure = make_structure(smiles, class_name)
    x, y, z = structure.coords[0]
    structure.coords[1] = (x + 0.3, y, z)
    return structure


def test_clash_free_structures_pass_and_a_duplicate_is_filtered(config):
    """Two entries that resolve to the same SMILES keep one and file the other."""
    first, second = make_structure(ALANINE, index=0), make_structure(ALANINE, index=1)
    output = repair(config, [first, second], {})
    assert [s.geometry_quality for s in output.structures] == ["ok"]
    assert [(r.reason, r.disposition) for r in output.rejected] == [
        ("dropped_duplicate", FILTERED)
    ]


def test_out_of_scope_organometallic_is_broken_with_geometry_unless_repair_is_off(config):
    structure = Structure(
        index=0,
        smiles="[Fe].c1ccccc1",
        class_name="planar",
        label=4,
        symbols=["Fe", "C"],
        coords=[(0.0, 0.0, 0.0), (0.4, 0.0, 0.0)],
        source_file="test.pkl",
    )
    output = repair(config, [structure], {})
    assert output.structures == []
    [rejected] = output.rejected
    assert (rejected.disposition, rejected.stage) == (BROKEN, "repair")
    assert rejected.has_geometry is True  # geometry survives for inspection

    disabled = config.model_copy(deep=True)
    disabled.repair.enabled = False
    output = repair(disabled, [structure], {})
    assert (len(output.structures), output.rejected) == (1, [])


def test_reembeddable_clash_is_repaired_and_marked(config):
    output = repair(config, [_collapsed(ALANINE, "central")], {})
    [repaired] = output.structures
    assert repaired.geometry_quality == "repaired"
    assert repaired.repair_strategy == "reembed"
    assert find_clashes(repaired.symbols, np.array(repaired.coords)) == []


def test_axial_clash_is_not_reembedded_because_smiles_cannot_encode_it(config):
    outcome = repair_structure(_collapsed("CC(C)(C)C(O)c1ccccc1", "axial"), config, {})
    assert outcome.status == "flagged"
    assert "not SMILES-encoded" in outcome.note
