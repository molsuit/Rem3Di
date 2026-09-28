"""The rebuild stage: fragment parsing, the frozen-geometry file and its gates.

The real frozen geometries are exercised by the integration test; here the
stage runs over a synthetic ferrocene written to a temporary frozen file.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml
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
CYMANTRENE = "[C-]#[O+].[C-]#[O+].[C-]#[O+].[CH]1[CH][CH][C]([CH]1)C.[Mn]"
ARENE_CHROMIUM = "[C-]#[O+].[C-]#[O+].[C-]#[O+].[CH]1[CH][CH][CH][C]([CH]1)C.[Cr]"


# --------------------------------------------------------------------------- #
# Ligand translation
# --------------------------------------------------------------------------- #


def test_ring_to_sandwich_makes_cp_anion_from_five_ring():
    ligand = ring_to_sandwich("[CH]1[CH][CH][CH][CH]1")
    assert ligand is not None
    mol = Chem.MolFromSmiles(ligand.smiles)
    assert mol is not None
    assert sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) == -1
    assert len(ligand.coord_list) == 5


def test_ring_to_sandwich_makes_neutral_arene_from_six_ring():
    ligand = ring_to_sandwich("[CH]1[CH][CH][CH][CH][CH]1")
    assert ligand is not None
    mol = Chem.MolFromSmiles(ligand.smiles)
    assert sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) == 0
    assert len(ligand.coord_list) == 6


def test_ring_to_sandwich_face_indices_are_all_ring_carbons():
    ligand = ring_to_sandwich("[CH]1[CH][CH][C]([CH]1)C=O")
    assert ligand is not None
    mol = Chem.MolFromSmiles(ligand.smiles)
    for index in ligand.coord_list:
        assert mol.GetAtomWithIdx(index).GetSymbol() == "C"
        assert mol.GetAtomWithIdx(index).IsInRing()


def test_ring_to_sandwich_preserves_substituents():
    ligand = ring_to_sandwich("[CH]1[CH][CH][C]([CH]1)C=O")
    assert ligand is not None
    mol = Chem.MolFromSmiles(ligand.smiles)
    assert any(atom.GetSymbol() == "O" for atom in mol.GetAtoms())


def test_ring_to_sandwich_returns_none_without_a_carbocyclic_face():
    assert ring_to_sandwich("CCO") is None


# --------------------------------------------------------------------------- #
# Complex parsing
# --------------------------------------------------------------------------- #


def test_parse_ferrocene():
    spec = parse_complex(FERROCENE)
    assert spec is not None
    assert spec.metal == "Fe"
    assert spec.n_carbon_monoxide == 0
    assert len(spec.ligands) == 2
    assert spec.family == "ferrocene (bis-Cp)"


def test_parse_cymantrene():
    spec = parse_complex(CYMANTRENE)
    assert spec is not None
    assert spec.metal == "Mn"
    assert spec.n_carbon_monoxide == 3
    assert len(spec.ligands) == 1
    assert spec.family == "Mn(CO)3 piano-stool"


def test_parse_arene_chromium_tricarbonyl():
    spec = parse_complex(ARENE_CHROMIUM)
    assert spec is not None
    assert spec.metal == "Cr"
    assert spec.n_carbon_monoxide == 3
    assert spec.family == "Cr(CO)3 piano-stool"


def test_parse_returns_none_for_unsupported_metal():
    assert parse_complex("[CH]1[CH][CH][CH][CH]1.[Ru]") is None


def test_parse_returns_none_when_a_fragment_is_not_a_ring():
    assert parse_complex("CCO.[Fe]") is None


# --------------------------------------------------------------------------- #
# Composition expectations used by the build validation gate
# --------------------------------------------------------------------------- #


def test_expected_counts_ferrocene_is_c10h10fe():
    spec = parse_complex(FERROCENE.replace("[C]([CH]1)C=O", "[CH]1"))
    if spec is None:  # the substituted variant; fall back to the plain rings
        spec = parse_complex("[CH]1[CH][CH][CH][CH]1.[CH]1[CH][CH][CH][CH]1.[Fe]")
    assert spec is not None
    counts = spec.expected_counts()
    assert counts["Fe"] == 1
    assert counts["C"] == 10
    assert counts["H"] == 10


def test_expected_counts_cymantrene_includes_the_carbonyls():
    spec = parse_complex(CYMANTRENE)
    assert spec is not None
    counts = spec.expected_counts()
    assert counts["Mn"] == 1
    assert counts["O"] == 3  # one per CO
    # 5 ring C + 1 methyl C + 3 carbonyl C
    assert counts["C"] == 9


def test_component_smiles_names_the_metal_and_every_carbonyl():
    chromium = parse_complex(ARENE_CHROMIUM)
    manganese = parse_complex(ARENE_CHROMIUM.replace("[Cr]", "[Mn]"))
    assert chromium is not None and manganese is not None
    assert chromium.component_smiles.endswith(".[C-]#[O+].[C-]#[O+].[C-]#[O+].[Cr]")
    assert chromium.component_smiles != manganese.component_smiles


def test_complex_spec_family_falls_back_for_unusual_carbonyl_counts():
    spec = ComplexSpec(metal="Fe", n_carbon_monoxide=2, ligands=[])
    assert spec.family == "Fe other"


# --------------------------------------------------------------------------- #
# Frozen rebuilds
# --------------------------------------------------------------------------- #

PLAIN_FERROCENE = "[CH]1[CH][CH][CH][CH]1.[CH]1[CH][CH][CH][CH]1.[Fe]"


def _ferrocene_geometry() -> FrozenRebuild:
    """An eclipsed ferrocene: two Cp rings 1.66 A above and below the iron."""
    symbols = ["Fe"]
    coords = [(0.0, 0.0, 0.0)]
    for height in (1.66, -1.66):
        for position in range(5):
            angle = 2 * math.pi * position / 5
            symbols.append("C")
            coords.append((1.21 * math.cos(angle), 1.21 * math.sin(angle), height))
        for position in range(5):
            angle = 2 * math.pi * position / 5
            symbols.append("H")
            coords.append((2.28 * math.cos(angle), 2.28 * math.sin(angle), height * 1.02))
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
    payload = {
        "extraction": {"data_dir": ".", "sources": []},
        "organometallic": {
            "rebuilt_structures": {"path": "frozen.extxyz", "sha256": sha256}
        },
    }
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(payload))
    return PipelineConfig.from_yaml(path)


def test_frozen_rebuilds_round_trip_through_the_extxyz_file(tmp_path):
    frozen = _ferrocene_geometry()
    _write_frozen(tmp_path / "frozen.extxyz", [frozen])
    read = read_frozen_rebuilds(tmp_path / "frozen.extxyz")
    assert list(read) == [PLAIN_FERROCENE]
    assert read[PLAIN_FERROCENE].symbols == frozen.symbols
    assert np.allclose(read[PLAIN_FERROCENE].coords, frozen.coords, atol=1e-6)


def test_duplicate_source_smiles_in_the_frozen_file_is_an_error(tmp_path):
    frozen = _ferrocene_geometry()
    _write_frozen(tmp_path / "frozen.extxyz", [frozen, frozen])
    with pytest.raises(ValueError, match="duplicate source_smiles"):
        read_frozen_rebuilds(tmp_path / "frozen.extxyz")


def test_a_valid_ferrocene_passes_every_gate():
    spec = parse_complex(PLAIN_FERROCENE)
    assert spec is not None
    assert validate_rebuild(spec, _ferrocene_geometry()) is None


def test_a_frozen_geometry_with_the_wrong_atoms_fails_composition():
    spec = parse_complex(PLAIN_FERROCENE)
    assert spec is not None
    frozen = _ferrocene_geometry()
    frozen.symbols = frozen.symbols[:-1]
    frozen.coords = frozen.coords[:-1]
    assert "composition" in (validate_rebuild(spec, frozen) or "")


def test_a_ring_lifted_off_the_metal_fails_hapticity():
    spec = parse_complex(PLAIN_FERROCENE)
    assert spec is not None
    frozen = _ferrocene_geometry()
    frozen.coords = [(x, y, z * 2.0) for x, y, z in frozen.coords]
    assert "haptically bound" in (validate_rebuild(spec, frozen) or "")


def test_rebuild_restores_targets_and_rejects_the_ones_without_geometry(tmp_path):
    _write_frozen(tmp_path / "frozen.extxyz", [_ferrocene_geometry()])
    config = _config(tmp_path)
    targets = [_target(PLAIN_FERROCENE, 0), _target(CYMANTRENE, 1)]
    output = rebuild(config, targets)
    assert [structure.index for structure in output.structures] == [0]
    assert output.structures[0].geometry_quality == "rebuilt_architector"
    assert [record.index for record in output.rejected] == [1]
    assert "no_frozen_geometry" in output.rejected[0].detail
    assert output.counts == {"targets": 2, "rebuilt": 1, "failed": 1}


def test_rebuild_refuses_a_frozen_file_that_is_not_the_pinned_one(tmp_path):
    _write_frozen(tmp_path / "frozen.extxyz", [_ferrocene_geometry()])
    with pytest.raises(ChecksumMismatchError):
        rebuild(_config(tmp_path, sha256="0" * 64), [_target(PLAIN_FERROCENE, 0)])
    pinned = sha256_of_file(tmp_path / "frozen.extxyz")
    assert rebuild(
        _config(tmp_path, sha256=pinned), [_target(PLAIN_FERROCENE, 0)]
    ).structures


def test_rebuild_fails_loudly_when_the_frozen_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Frozen organometallic rebuilds"):
        rebuild(_config(tmp_path), [_target(PLAIN_FERROCENE, 0)])
