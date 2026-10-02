"""Tests for the QM9-OR preparer.

Everything except the last test runs on a tiny synthetic ``qm9-or.npy`` in the
source layout (dict entries with ``index``, ``inchi``, padded ``xyz``,
``chiral_centers``, ``rotation``); the last test rebuilds the real table against
its pins and is skipped when the 237 MB download is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from pydantic import ValidationError
from rdkit import Chem
from remedi_prepare_qm9or.prepare import (
    DEFAULT_RAW_ROOT,
    MOLECULE_COLUMNS,
    ONE_HOT_ATOMIC_NUMBERS,
    QM9OR_PINS,
    QM9OR_SOURCE,
    XYZ_SHAPE,
    QM9ORPreparationError,
    QM9ORPreparerConfig,
    RandomSplit,
    ScaffoldSplit,
    SourceFile,
    SplitCounts,
    TablePins,
    build_dataset,
    build_dataset_from_entries,
    collapse_entries,
    compare_composition,
    decode_geometry,
    download_source,
    ensure_source,
    entry_labels,
    file_digests,
    format_summary,
    main,
    random_split_values,
    scaffold_groups,
    scaffold_split_values,
    table_hashes,
    verify_source_file,
    verify_table,
)

R_ALANINE = "C[C@@H](N)C(=O)O"
S_ALANINE = "C[C@H](N)C(=O)O"
ETHANOL = "CCO"
R_R_BUTANEDIOL = "C[C@@H](O)[C@@H](C)O"
PROPANOL = "CCCO"
BENZENE = "c1ccccc1"
TOLUENE = "Cc1ccccc1"
CYCLOHEXANOL = "OC1CCCCC1"
PYRIDINE = "c1ccncc1"


def padded_xyz(smiles: str, drop_hydrogens: int = 0) -> np.ndarray:
    """A ``(27, 8)`` array for ``smiles`` with explicit hydrogens, zero-padded.

    ``drop_hydrogens`` removes that many trailing hydrogens, imitating the
    QM9-OR entries whose geometry and InChI disagree on the hydrogen count.
    """
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    for _ in range(drop_hydrogens):
        numbers.remove(1)
    array = np.zeros(XYZ_SHAPE)
    for row, number in enumerate(numbers):
        array[row, :3] = (row + 0.5, -row, 2.0 * row)
        array[row, 3 + ONE_HOT_ATOMIC_NUMBERS.index(number)] = 1.0
    return array


def make_entry(
    smiles: str,
    index: str,
    rotation_589: float,
    centres: list[tuple[int, str]] | None = None,
    drop_hydrogens: int = 0,
) -> dict[str, Any]:
    return {
        "index": index,
        "inchi": Chem.MolToInchi(Chem.MolFromSmiles(smiles)),
        "xyz": padded_xyz(smiles, drop_hydrogens),
        "chiral_centers": centres or [],
        "rotation": [rotation_589 * 0.5, rotation_589, rotation_589 * 2.0],
    }


def synthetic_entries() -> list[dict[str, Any]]:
    """Nine entries, eight molecules: S-alanine duplicates R-alanine (opposite sign)."""
    return [
        make_entry(R_ALANINE, "000001", 12.5, [(1, "R")]),
        make_entry(ETHANOL, "000002", -0.0),
        make_entry(S_ALANINE, "000003", -12.5, [(1, "S")]),
        make_entry(R_R_BUTANEDIOL, "000004", -3.1, [(1, "R"), (2, "R")]),
        make_entry(PROPANOL, "000005", 0.01, [(1, "s")]),
        make_entry(BENZENE, "000006", 0.0),
        make_entry(TOLUENE, "000007", -0.02, drop_hydrogens=1),
        make_entry(CYCLOHEXANOL, "000008", 4.0),
        make_entry(PYRIDINE, "000009", -7.0),
    ]


def as_object_array(entries: list[dict[str, Any]]) -> np.ndarray:
    array = np.empty(len(entries), dtype=object)
    array[:] = entries
    return array


def unpinned_config(
    raw_root: Path, source: SourceFile | None = None
) -> QM9ORPreparerConfig:
    return QM9ORPreparerConfig(
        raw_root=raw_root,
        source=source or QM9OR_SOURCE,
        download=False,
        pins=None,
    )


def write_synthetic_source(directory: Path) -> SourceFile:
    """Save the synthetic entries as ``qm9-or.npy``; return a matching pin."""
    path = directory / "qm9-or.npy"
    np.save(path, as_object_array(synthetic_entries()), allow_pickle=True)
    sha256, md5 = file_digests(path)
    return SourceFile(
        url=path.as_uri(),
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=sha256,
        md5=md5,
    )


# ------------------------------------------------------------- geometry


def test_decode_geometry_strips_padding() -> None:
    atoms = decode_geometry(padded_xyz(ETHANOL), position=0)
    assert isinstance(atoms, Atoms)
    assert len(atoms) == 9
    assert atoms.get_chemical_formula() == "C2H6O"
    np.testing.assert_allclose(atoms.positions[1], (1.5, -1.0, 2.0))


def test_decode_geometry_rejects_padding_between_atoms() -> None:
    array = padded_xyz(ETHANOL)
    array[[1, 20]] = array[[20, 1]]
    with pytest.raises(QM9ORPreparationError, match="padding rows must follow"):
        decode_geometry(array, position=3)


def test_decode_geometry_rejects_nonzero_padding() -> None:
    array = padded_xyz(ETHANOL)
    array[-1, 0] = 0.25
    with pytest.raises(QM9ORPreparationError, match="not all zero"):
        decode_geometry(array, position=3)


def test_decode_geometry_rejects_ambiguous_atom_type() -> None:
    array = padded_xyz(ETHANOL)
    array[0, 3:] = (1.0, 1.0, 0.0, 0.0, 0.0)
    with pytest.raises(QM9ORPreparationError, match="not one-hot"):
        decode_geometry(array, position=3)


def test_compare_composition_counts_hydrogens_separately() -> None:
    molecule = Chem.MolFromSmiles(TOLUENE)
    agreeing = compare_composition(decode_geometry(padded_xyz(TOLUENE), 0), molecule)
    assert agreeing.heavy_atoms_agree and agreeing.hydrogens_agree
    short = compare_composition(
        decode_geometry(padded_xyz(TOLUENE, drop_hydrogens=2), 0), molecule
    )
    assert short.heavy_atoms_agree
    assert (short.geometry_hydrogens, short.molecule_hydrogens) == (6, 8)


def test_heavy_atom_disagreement_is_an_error() -> None:
    entry = make_entry(ETHANOL, "000001", 1.0)
    entry["xyz"] = padded_xyz(PROPANOL)
    with pytest.raises(QM9ORPreparationError, match="heavy atoms"):
        collapse_entries([entry])


def test_entry_layout_is_checked() -> None:
    entry = make_entry(ETHANOL, "000001", 1.0)
    del entry["rotation"]
    with pytest.raises(QM9ORPreparationError, match="expected keys"):
        collapse_entries([entry])


# --------------------------------------------------------------- labels


@pytest.mark.parametrize(
    ("centres", "rotation_589", "expected"),
    [
        ([], -0.0, (0, 0, 0)),
        ([(1, "S")], 0.01, (1, 1, 1)),
        ([(1, "R")], -0.01, (1, 0, -1)),
        ([(1, "s")], 5.0, (1, 0, 1)),
        ([(0, "S"), (2, "S")], -3.0, (2, 0, -1)),
        ([(0, "S"), (1, "r"), (2, "R")], 0.0, (3, 0, 0)),
    ],
)
def test_entry_labels(
    centres: list[tuple[int, str]], rotation_589: float, expected: tuple[int, int, int]
) -> None:
    labels = entry_labels(
        {"chiral_centers": centres, "rotation": [9.0, rotation_589, 9.0]}
    )
    assert (labels.n_chiral, labels.rs, labels.or_sign_589) == expected


def test_collapse_deduplicates_on_nonisomeric_smiles_first_entry_wins() -> None:
    collapsed = collapse_entries(synthetic_entries())
    table = pd.DataFrame(collapsed.rows)
    assert collapsed.source_entry_count == 9
    assert list(table.molecule_id) == list(range(8))
    # S-alanine (third entry) folds into R-alanine, whose labels are kept.
    first = table.iloc[0]
    assert first.isomeric_smiles == Chem.MolToSmiles(Chem.MolFromSmiles(R_ALANINE))
    assert (first.n_chiral, first.rs, first.or_sign_589) == (1, 0, 1)
    assert collapsed.signs_per_molecule[0] == [1, -1]
    assert table.isomeric_smiles.tolist()[1] == "CCO"
    assert table.or_sign_589.tolist() == [1, 0, -1, 1, 0, -1, 1, -1]
    assert table.n_chiral.tolist() == [1, 0, 2, 1, 0, 0, 0, 0]
    assert table.rs.tolist() == [0] * 8


def test_frames_align_with_rows() -> None:
    collapsed = collapse_entries(synthetic_entries())
    assert len(collapsed.frames) == len(collapsed.rows)
    for row, atoms in zip(collapsed.rows, collapsed.frames, strict=True):
        assert atoms.info["molecule_id"] == row["molecule_id"]
        molecule = Chem.AddHs(Chem.MolFromSmiles(row["isomeric_smiles"]))
        heavy = sorted(
            atom.GetAtomicNum()
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() > 1
        )
        assert sorted(int(number) for number in atoms.numbers if number > 1) == heavy
    # The pyridine row comes from the ninth entry: the duplicate shifted it by one.
    assert collapsed.frames[-1].info == {
        "molecule_id": 7,
        "qm9_index": "000009",
        "source_entry": 8,
    }
    # Toluene's geometry lacks a hydrogen.
    assert collapsed.hydrogen_mismatch_molecule_ids == [5]


# --------------------------------------------------------------- splits


def test_random_split_counts_and_determinism() -> None:
    split = RandomSplit(seed=42)
    values = random_split_values(117, split)
    assert (values == random_split_values(117, split)).all()
    assert not (values == random_split_values(117, RandomSplit(seed=43))).all()
    assert {
        name: int((values == name).sum()) for name in ("train", "valid", "test")
    } == {
        "train": 82,
        "valid": 12,
        "test": 23,
    }


def test_scaffold_split_keeps_groups_whole_and_is_deterministic() -> None:
    smiles = [
        BENZENE,
        TOLUENE,
        ETHANOL,
        PROPANOL,
        CYCLOHEXANOL,
        PYRIDINE,
        "Oc1ccccc1",
        "CC",
    ]
    groups = scaffold_groups(smiles, include_chirality=True)
    # benzene/toluene/phenol share a scaffold, as do the acyclic molecules.
    assert sorted(map(sorted, groups)) == [[0, 1, 6], [2, 3, 7], [4], [5]]
    split = ScaffoldSplit(seed=0, train_fraction=0.5, valid_fraction=0.25)
    values = scaffold_split_values(groups, len(smiles), split)
    assert (values == scaffold_split_values(groups, len(smiles), split)).all()
    for group in groups:
        assert len(set(values[group])) == 1
    assert int((values == "train").sum()) <= 4
    assert set(values) <= {"train", "valid", "test"}


def test_split_definitions_from_yaml_like_data() -> None:
    config = QM9ORPreparerConfig.model_validate(
        {
            "pins": None,
            "splits": [
                {"kind": "random", "seed": 7, "test_fraction": 0.3},
                {"kind": "scaffold", "seed": 1, "include_chirality": False},
            ],
        }
    )
    assert [split.column_name for split in config.splits] == [
        "random_s7",
        "scaffold_s1",
    ]
    assert isinstance(config.splits[1], ScaffoldSplit)


def test_config_rejects_duplicate_columns_and_unpinned_splits() -> None:
    with pytest.raises(ValidationError, match="duplicate split columns"):
        QM9ORPreparerConfig(
            pins=None, splits=[RandomSplit(seed=1), RandomSplit(seed=1)]
        )
    with pytest.raises(ValidationError, match="pins cover"):
        QM9ORPreparerConfig(splits=[RandomSplit(seed=1)])
    with pytest.raises(ValidationError, match="leave a train fold"):
        RandomSplit(seed=1, test_fraction=0.6, valid_fraction=0.4)


def test_default_config_is_the_frozen_table() -> None:
    config = QM9ORPreparerConfig()
    assert config.raw_root == DEFAULT_RAW_ROOT
    assert [split.column_name for split in config.splits] == [
        "random_s42",
        "random_s43",
        "random_s44",
        "random_s45",
        "scaffold_s0",
        "scaffold_s1",
        "scaffold_s2",
    ]
    assert config.pins == QM9OR_PINS


# --------------------------------------------------------- verification


def synthetic_dataset_and_pins() -> tuple[pd.DataFrame, TablePins]:
    config = QM9ORPreparerConfig(
        pins=None, splits=[RandomSplit(seed=1), ScaffoldSplit(seed=0)]
    )
    dataset = build_dataset_from_entries(synthetic_entries(), config)
    pins = TablePins(
        molecule_count=dataset.summary.molecules,
        hashes=dataset.summary.hashes,
        split_counts=dataset.summary.split_counts,
    )
    return dataset.table, pins


def test_verify_table_accepts_its_own_pins() -> None:
    table, pins = synthetic_dataset_and_pins()
    verify_table(table, table_hashes(table), pins)
    assert list(table.columns) == [*MOLECULE_COLUMNS, "random_s1", "scaffold_s0"]


def test_verify_table_blames_rdkit_when_only_smiles_differ() -> None:
    table, pins = synthetic_dataset_and_pins()
    changed = table.copy()
    changed.loc[1, "isomeric_smiles"] = "OCC"
    with pytest.raises(
        QM9ORPreparationError, match="only the SMILES strings differ"
    ) as error:
        verify_table(changed, table_hashes(changed), pins)
    assert "molecule columns sha256" in str(error.value)


def test_verify_table_reports_other_differences() -> None:
    table, pins = synthetic_dataset_and_pins()
    changed = table.copy()
    changed.loc[0, "random_s1"] = (
        "test" if changed.loc[0, "random_s1"] != "test" else "train"
    )
    with pytest.raises(
        QM9ORPreparationError, match="other than isomeric_smiles"
    ) as error:
        verify_table(changed, table_hashes(changed), pins)
    assert "random_s1 counts" in str(error.value)


def test_verify_table_checks_molecule_count() -> None:
    table, pins = synthetic_dataset_and_pins()
    wrong = pins.model_copy(update={"molecule_count": 9})
    with pytest.raises(QM9ORPreparationError, match="8 molecules, expected 9"):
        verify_table(table, table_hashes(table), wrong)


def test_split_counts_pin_mismatch() -> None:
    table, pins = synthetic_dataset_and_pins()
    counts = dict(pins.split_counts)
    counts["scaffold_s0"] = SplitCounts(train=0, valid=0, test=8)
    with pytest.raises(QM9ORPreparationError, match="scaffold_s0 counts"):
        verify_table(
            table, table_hashes(table), pins.model_copy(update={"split_counts": counts})
        )


# ------------------------------------------------------------ the source


def test_verify_source_file(tmp_path: Path) -> None:
    source = write_synthetic_source(tmp_path)
    verify_source_file(tmp_path / "qm9-or.npy", source)
    wrong = source.model_copy(update={"md5": "0" * 32, "sha256": "f" * 64})
    with pytest.raises(QM9ORPreparationError, match=r"sha256 .* md5"):
        verify_source_file(tmp_path / "qm9-or.npy", wrong)
    with pytest.raises(QM9ORPreparationError, match="not found"):
        verify_source_file(tmp_path / "missing.npy", source)


def test_download_verifies_before_publishing(tmp_path: Path) -> None:
    source = write_synthetic_source(tmp_path)
    destination = tmp_path / "raw" / "qm9-or.npy"
    download_source(source, destination)
    assert file_digests(destination) == (source.sha256, source.md5)
    assert not destination.with_name("qm9-or.npy.part").exists()

    corrupt = tmp_path / "corrupt" / "qm9-or.npy"
    with pytest.raises(QM9ORPreparationError, match="sha256"):
        download_source(source.model_copy(update={"sha256": "a" * 64}), corrupt)
    assert not corrupt.exists()
    assert not corrupt.with_name("qm9-or.npy.part").exists()

    with pytest.raises(QM9ORPreparationError, match="download of"):
        download_source(
            source.model_copy(update={"url": (tmp_path / "absent.npy").as_uri()}),
            corrupt,
        )


def test_ensure_source_downloads_only_when_allowed(tmp_path: Path) -> None:
    source = write_synthetic_source(tmp_path)
    raw_root = tmp_path / "raw"
    with pytest.raises(QM9ORPreparationError, match="downloading is disabled"):
        ensure_source(unpinned_config(raw_root, source))
    allowed = unpinned_config(raw_root, source).model_copy(update={"download": True})
    assert ensure_source(allowed) == raw_root / "qm9-or.npy"


def test_build_dataset_end_to_end_on_synthetic_source(tmp_path: Path) -> None:
    source = write_synthetic_source(tmp_path)
    config = QM9ORPreparerConfig(
        raw_root=tmp_path,
        source=source,
        download=False,
        pins=None,
        splits=[RandomSplit(seed=42), ScaffoldSplit(seed=0)],
    )
    dataset = build_dataset(config)
    summary = dataset.summary
    assert (summary.source_entries, summary.molecules, summary.duplicate_entries) == (
        9,
        8,
        1,
    )
    assert summary.molecules_with_duplicates == 1
    assert summary.opposite_sign_molecules == summary.differing_sign_molecules == 1
    assert dataset.opposite_sign_molecule_ids == [0]
    assert summary.hydrogen_mismatch_molecules == 1
    assert summary.split_counts["random_s42"] == SplitCounts(train=5, valid=1, test=2)
    assert not summary.verified
    assert len(dataset.frames) == len(dataset.table) == 8
    assert "NOT verified" in format_summary(summary)

    pinned = config.model_copy(
        update={
            "pins": TablePins(
                molecule_count=8,
                hashes=summary.hashes,
                split_counts=summary.split_counts,
            )
        }
    )
    again = build_dataset(pinned)
    assert again.summary.verified
    pd.testing.assert_frame_equal(again.table, dataset.table)


def test_main_reports_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--raw-root", str(tmp_path), "--no-download"]) == 1
    assert capsys.readouterr().out == ""


# ------------------------------------------------------ the real download


REAL_SOURCE = DEFAULT_RAW_ROOT / QM9OR_SOURCE.filename


@pytest.mark.skipif(not REAL_SOURCE.is_file(), reason=f"{REAL_SOURCE} not downloaded")
def test_real_rebuild_matches_pins() -> None:
    dataset = build_dataset(QM9ORPreparerConfig(download=False))
    summary = dataset.summary
    assert summary.verified
    assert summary.hashes == QM9OR_PINS.hashes
    assert (summary.source_entries, summary.molecules) == (121_416, 117_625)
    assert summary.opposite_sign_molecules == 1_550
    assert summary.differing_sign_molecules == 1_681
    assert summary.hydrogen_mismatch_molecules == 1_514
    assert summary.scaffold_groups == 19_074
    assert len(dataset.frames) == 117_625
    assert all(
        atoms.info["molecule_id"] == row for row, atoms in enumerate(dataset.frames)
    )
