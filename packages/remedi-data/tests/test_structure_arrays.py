"""A dataset's structures as ``remedi_data`` exposes them to training code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from remedi_data.dataset import Dataset, Structures, write_structures
from remedi_data.format import STRUCTURES_FILENAME, BundleValidationError
from testing import (
    ACHIRAL_TEN_SMILES,
    embed_frame,
    write_ten_structure_dataset,
)

#: What training reads from ``Structures``.
STRUCTURE_ARRAYS = (
    "ptr",
    "positions",
    "atomic_numbers",
    "total_charge",
    "multiplicity",
)


@pytest.fixture
def frames() -> list[Atoms]:
    return [embed_frame(smiles) for smiles in ACHIRAL_TEN_SMILES]


@pytest.fixture
def ten_structure_dataset(tmp_path: Path, frames: list[Atoms]) -> Path:
    return write_ten_structure_dataset(tmp_path, frames)


def test_a_dataset_is_the_table_files_and_one_structure_file(
    ten_structure_dataset: Path,
) -> None:
    assert sorted(path.name for path in ten_structure_dataset.iterdir()) == [
        "dataset.yaml",
        "provenance.yaml",
        STRUCTURES_FILENAME,
        "table.parquet",
    ]


def test_the_structures_hold_the_written_frames_and_the_table_charges(
    ten_structure_dataset: Path, frames: list[Atoms]
) -> None:
    dataset = Dataset.open(ten_structure_dataset)
    assert len(dataset) == len(frames)
    structures = dataset.read_structures()
    assert structures.ptr.tolist() == np.cumsum([0] + [len(f) for f in frames]).tolist()
    for index, frame in enumerate(frames):
        numbers, positions = structures.frame(index)
        np.testing.assert_array_equal(positions, frame.positions.astype(np.float32))
        np.testing.assert_array_equal(numbers, frame.numbers.astype(np.uint8))
        file_numbers, file_positions = dataset.structures.frame(index)
        np.testing.assert_array_equal(file_positions, positions)
        np.testing.assert_array_equal(file_numbers, numbers)
    np.testing.assert_array_equal(structures.total_charge, np.zeros(10, np.float32))
    np.testing.assert_array_equal(structures.multiplicity, np.ones(10, np.float32))


def test_a_selection_is_gathered_and_renumbered(
    ten_structure_dataset: Path, frames: list[Atoms]
) -> None:
    dataset = Dataset.open(ten_structure_dataset)
    chosen = np.array([1, 4, 5, 9])
    gathered = dataset.read_structures(chosen)
    expected = dataset.read_structures().take(chosen)
    for attribute in STRUCTURE_ARRAYS:
        np.testing.assert_array_equal(
            getattr(gathered, attribute), getattr(expected, attribute)
        )
    assert gathered.atom_counts.tolist() == [len(frames[i]) for i in chosen]
    assert (
        dataset.structures.atom_counts(chosen).tolist() == gathered.atom_counts.tolist()
    )


@pytest.mark.parametrize("structure_ids", [[2, 1], [1, 1], [-1], [10]])
def test_a_selection_must_be_sorted_unique_and_in_range(
    ten_structure_dataset: Path, structure_ids: list[int]
) -> None:
    dataset = Dataset.open(ten_structure_dataset)
    with pytest.raises(ValueError, match="structure_ids"):
        dataset.read_structures(np.array(structure_ids))


def test_a_directory_without_a_structure_file_is_refused(
    ten_structure_dataset: Path,
) -> None:
    (ten_structure_dataset / STRUCTURES_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match=STRUCTURES_FILENAME):
        Dataset.open(ten_structure_dataset)


def test_a_structure_file_with_another_record_count_is_refused(
    ten_structure_dataset: Path, frames: list[Atoms]
) -> None:
    (ten_structure_dataset / STRUCTURES_FILENAME).unlink()
    write_structures(
        ten_structure_dataset / STRUCTURES_FILENAME,
        [Structures.from_atoms(frames[:9])],
    )
    with pytest.raises(BundleValidationError, match="9 structures for 10 table rows"):
        Dataset.open(ten_structure_dataset)


def test_the_table_is_read_whole_or_by_column(ten_structure_dataset: Path) -> None:
    dataset = Dataset.open(ten_structure_dataset)
    assert dataset.table().columns[:2] == ["structure_id", "stereoisomer_id"]
    assert dataset.table(["y"]).columns == ["y"]
    assert dataset.spec.dataset_id == "ten"


def test_charges_follow_take_and_concatenate() -> None:
    structures = Structures.from_frames(
        [(np.array([6]), np.zeros((1, 3))), (np.array([8, 1]), np.ones((2, 3)))]
    ).with_charges(np.array([0.0, -1.0]), np.array([1.0, 2.0]))
    assert structures.total_charge is not None
    assert structures.total_charge.dtype == np.float32
    taken = structures.take([1])
    assert taken.total_charge is not None and taken.multiplicity is not None
    assert (taken.total_charge.tolist(), taken.multiplicity.tolist()) == ([-1.0], [2.0])
    joined = Structures.concatenate([structures, taken])
    assert joined.total_charge is not None
    assert joined.total_charge.tolist() == [0.0, -1.0, -1.0]
    with pytest.raises(ValueError, match="only some parts"):
        Structures.concatenate([structures, Structures.empty()])
    with pytest.raises(ValueError, match="total_charge has shape"):
        structures.with_charges(np.zeros(3), np.ones(2))


def test_to_atoms_carries_charge_and_multiplicity_only_when_present() -> None:
    frames = Structures.from_frames(
        [
            (np.array([8, 1, 1]), np.arange(9.0).reshape(3, 3)),
            (np.array([26]), np.zeros((1, 3))),
        ]
    )
    without = frames.to_atoms()
    assert [atoms.get_chemical_symbols() for atoms in without] == [
        ["O", "H", "H"],
        ["Fe"],
    ]
    assert all(atoms.info == {} and not atoms.pbc.any() for atoms in without)
    with_charges = frames.with_charges([0.0, 2.0], [1.0, 5.0]).to_atoms()
    assert [atoms.info for atoms in with_charges] == [
        {"total_charge": 0.0, "multiplicity": 1.0},
        {"total_charge": 2.0, "multiplicity": 5.0},
    ]
    assert np.allclose(with_charges[0].get_positions(), np.arange(9.0).reshape(3, 3))
