"""The rebuild stage: fragment parsing, the frozen-geometry file and its gates.

The real frozen geometries are exercised by the integration test; here the
stage runs over a synthetic ferrocene written to a temporary frozen file.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import write_config
from rdkit import Chem

from chiralcat_dataset.config import PipelineConfig
from chiralcat_dataset.organometallic import (
    ComplexSpec,
    FrozenRebuild,
    parse_complex,
    read_frozen_rebuilds,
    rebuild,
    ring_to_sandwich,
    validate_rebuild,
)
from chiralcat_dataset.records import BROKEN, RejectedRecord
from chiralcat_dataset.sources import ChecksumMismatchError, sha256_of_file
from chiralcat_dataset.writers import _comment_line

FERROCENE = "[CH]1[CH][CH][C]([CH]1)C=O.[CH]1[CH][CH][CH][CH]1.[Fe]"
PLAIN_FERROCENE = "[CH]1[CH][CH][CH][CH]1.[CH]1[CH][CH][CH][CH]1.[Fe]"
CYMANTRENE = "[C-]#[O+].[C-]#[O+].[C-]#[O+].[CH]1[CH][CH][C]([CH]1)C.[Mn]"
ARENE_CHROMIUM = "[C-]#[O+].[C-]#[O+].[C-]#[O+].[CH]1[CH][CH][CH][C]([CH]1)C.[Cr]"


@pytest.mark.parametrize(
    ("fragment", "charge", "face_size"),
    [
        ("[CH]1[CH][CH][CH][CH]1", -1, 5),  # Cp anion
        ("[CH]1[CH][CH][CH][CH][CH]1", 0, 6),  # neutral arene
        ("[CH]1[CH][CH][C]([CH]1)C=O", -1, 5),  # substituent kept off the face
    ],
)
def test_ring_to_sandwich_aromatizes_the_haptic_face(fragment, charge, face_size):
    ligand = ring_to_sandwich(fragment)
    assert ligand is not None
    mol = Chem.MolFromSmiles(ligand.smiles)
    assert sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) == charge
    assert len(ligand.coord_list) == face_size
    for index in ligand.coord_list:
        atom = mol.GetAtomWithIdx(index)
        assert atom.GetSymbol() == "C" and atom.IsInRing()
    assert mol.GetNumHeavyAtoms() == Chem.MolFromSmiles(fragment).GetNumHeavyAtoms()


@pytest.mark.parametrize(
    ("smiles", "family", "n_ligands", "expected_counts"),
    [
        (FERROCENE, "ferrocene (bis-Cp)", 2, {"Fe": 1, "C": 11, "H": 10, "O": 1}),
        # one C and one O per carbonyl
        (CYMANTRENE, "Mn(CO)3 piano-stool", 1, {"Mn": 1, "C": 9, "H": 7, "O": 3}),
        (ARENE_CHROMIUM, "Cr(CO)3 piano-stool", 1, {"Cr": 1, "C": 10, "H": 8, "O": 3}),
    ],
)
def test_parse_complex_recognises_the_three_scaffold_families(
    smiles, family, n_ligands, expected_counts
):
    spec = parse_complex(smiles)
    assert spec is not None
    assert spec.family == family
    assert len(spec.ligands) == n_ligands
    assert spec.expected_counts() == expected_counts


def test_parse_complex_refuses_what_it_cannot_build():
    assert ring_to_sandwich("CCO") is None  # no carbocyclic face
    assert parse_complex("[CH]1[CH][CH][CH][CH]1.[Ru]") is None  # unsupported metal
    assert parse_complex("CCO.[Fe]") is None  # a fragment that is not a ring
    assert parse_complex("CCO") is None  # no metal at all


def test_complex_spec_family_falls_back_for_unusual_carbonyl_counts():
    assert ComplexSpec(metal="Fe", n_carbon_monoxide=2, ligands=[]).family == "Fe other"


def test_component_smiles_names_the_metal_and_every_carbonyl():
    """Regression: an (arene)Cr(CO)3 and (arene)Mn(CO)3 once shared a SMILES."""
    chromium = parse_complex(ARENE_CHROMIUM)
    manganese = parse_complex(ARENE_CHROMIUM.replace("[Cr]", "[Mn]"))
    assert chromium is not None and manganese is not None
    assert chromium.component_smiles.endswith(".[C-]#[O+].[C-]#[O+].[C-]#[O+].[Cr]")
    assert chromium.component_smiles != manganese.component_smiles


# --------------------------------------------------------------------------- #
# Frozen rebuilds
# --------------------------------------------------------------------------- #


def _ferrocene_geometry() -> FrozenRebuild:
    """An eclipsed ferrocene: two Cp rings 1.66 A above and below the iron."""
    symbols = ["Fe"]
    coords = [(0.0, 0.0, 0.0)]
    for height in (1.66, -1.66):
        for symbol, radius, lift in (("C", 1.21, 1.0), ("H", 2.28, 1.02)):
            for position in range(5):
                angle = 2 * math.pi * position / 5
                symbols.append(symbol)
                coords.append(
                    (radius * math.cos(angle), radius * math.sin(angle), height * lift)
                )
    return FrozenRebuild(PLAIN_FERROCENE, symbols, coords)


def _write_frozen(path, rebuilds):
    with open(path, "w", encoding="utf-8") as handle:
        for frozen in rebuilds:
            handle.write(f"{len(frozen.symbols)}\n")
            handle.write(_comment_line({"source_smiles": frozen.source_smiles}) + "\n")
            handle.writelines(
                f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n"
                for symbol, (x, y, z) in zip(frozen.symbols, frozen.coords, strict=True)
            )


def _target(smiles: str, index: int) -> RejectedRecord:
    return RejectedRecord(
        smiles=smiles,
        class_name="planar",
        label=4,
        source_file="synthetic.pkl",
        stage="repair",
        reason="flagged",
        disposition=BROKEN,
        index=index,
    )


def _config(tmp_path, sha256=None) -> PipelineConfig:
    rebuilt_structures = {"path": "frozen.extxyz", "sha256": sha256}
    path = write_config(
        tmp_path,
        extraction={"data_dir": "."},
        organometallic={"enabled": True, "rebuilt_structures": rebuilt_structures},
    )
    return PipelineConfig.from_yaml(path)


def test_frozen_rebuilds_round_trip_and_refuse_a_duplicate_key(tmp_path):
    frozen = _ferrocene_geometry()
    _write_frozen(tmp_path / "frozen.extxyz", [frozen])
    read = read_frozen_rebuilds(tmp_path / "frozen.extxyz")
    assert list(read) == [PLAIN_FERROCENE]
    assert read[PLAIN_FERROCENE].symbols == frozen.symbols
    assert np.allclose(read[PLAIN_FERROCENE].coords, frozen.coords, atol=1e-6)

    _write_frozen(tmp_path / "frozen.extxyz", [frozen, frozen])
    with pytest.raises(ValueError, match="duplicate source_smiles"):
        read_frozen_rebuilds(tmp_path / "frozen.extxyz")


def _drop_last_atom(frozen: FrozenRebuild) -> None:
    frozen.symbols, frozen.coords = frozen.symbols[:-1], frozen.coords[:-1]


def _lift_the_rings(frozen: FrozenRebuild) -> None:
    frozen.coords = [(x, y, z * 2.0) for x, y, z in frozen.coords]


def _stack_two_hydrogens(frozen: FrozenRebuild) -> None:
    frozen.coords[-1] = frozen.coords[-2]


@pytest.mark.parametrize(
    ("distort", "failure"),
    [
        (None, None),
        (_drop_last_atom, "composition"),
        (_lift_the_rings, "haptically bound"),
        (_stack_two_hydrogens, "residual clash"),
    ],
)
def test_validate_rebuild_gates(distort, failure):
    spec = parse_complex(PLAIN_FERROCENE)
    assert spec is not None
    frozen = _ferrocene_geometry()
    if distort is not None:
        distort(frozen)
    result = validate_rebuild(spec, frozen)
    if failure is None:
        assert result is None
    else:
        assert failure in (result or "")


def test_rebuild_restores_targets_and_rejects_the_ones_without_geometry(tmp_path):
    _write_frozen(tmp_path / "frozen.extxyz", [_ferrocene_geometry()])
    targets = [_target(PLAIN_FERROCENE, 0), _target(CYMANTRENE, 1)]
    output = rebuild(_config(tmp_path), targets)
    assert [structure.index for structure in output.structures] == [0]
    assert output.structures[0].geometry_quality == "rebuilt_architector"
    assert [record.index for record in output.rejected] == [1]
    assert "no_frozen_geometry" in output.rejected[0].detail
    assert output.counts == {"targets": 2, "rebuilt": 1, "failed": 1}


def test_rebuild_refuses_a_frozen_file_that_is_not_the_pinned_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="Frozen organometallic rebuilds"):
        rebuild(_config(tmp_path), [_target(PLAIN_FERROCENE, 0)])
    _write_frozen(tmp_path / "frozen.extxyz", [_ferrocene_geometry()])
    with pytest.raises(ChecksumMismatchError):
        rebuild(_config(tmp_path, sha256="0" * 64), [_target(PLAIN_FERROCENE, 0)])
    pinned = sha256_of_file(tmp_path / "frozen.extxyz")
    assert rebuild(
        _config(tmp_path, sha256=pinned), [_target(PLAIN_FERROCENE, 0)]
    ).structures
