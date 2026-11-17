"""The zarr structure arrays as ``remedi_data`` exposes them to training code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from ase import Atoms

from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.dataset.packed_structures import PackedStructures
from testing import (
    ACHIRAL_TEN_SMILES,
    embed_frame,
    write_ten_structure_dataset,
)

#: What training reads: the attributes of ``MoleculeDataset`` and
#: ``PackedStructures``, and the array names in the zarr group.
STRUCTURE_ARRAYS = {
    "ptr": "molecule_ptr",
    "positions": "positions",
    "atomic_numbers": "atomic_numbers",
    "total_charge": "total_charge",
    "multiplicity": "multiplicity",
}


@pytest.fixture
def frames() -> list[Atoms]:
    return [embed_frame(smiles) for smiles in ACHIRAL_TEN_SMILES]


@pytest.fixture
def ten_structure_dataset(tmp_path: Path, frames: list[Atoms]) -> Path:
    return write_ten_structure_dataset(tmp_path, frames)


def test_the_structure_arrays_are_in_the_zarr(ten_structure_dataset: Path) -> None:
    group = zarr.open_group(ten_structure_dataset, mode="r")
    for array_name in STRUCTURE_ARRAYS.values():
        assert array_name in group, f"{array_name} is not an array in the zarr"


@pytest.mark.parametrize("in_memory", [True, False])
def test_the_structure_arrays_hold_the_written_structures(
    ten_structure_dataset: Path, frames: list[Atoms], in_memory: bool
) -> None:
    dataset = MoleculeDataset.open_existing_dataset_from_dir(ten_structure_dataset)
    arrays = dataset.read_structures() if in_memory else dataset
    for attribute in STRUCTURE_ARRAYS:
        assert isinstance(getattr(arrays, attribute), np.ndarray) is in_memory

    pointer = np.asarray(arrays.ptr[:])
    assert pointer.tolist() == np.cumsum([0] + [len(f) for f in frames]).tolist()
    for index, frame in enumerate(frames):
        start, end = pointer[index], pointer[index + 1]
        np.testing.assert_array_equal(
            arrays.positions[start:end], frame.positions.astype(np.float32)
        )
        np.testing.assert_array_equal(
            arrays.atomic_numbers[start:end], frame.numbers.astype(np.uint8)
        )
    np.testing.assert_array_equal(arrays.total_charge[:], np.zeros(10, np.float32))
    np.testing.assert_array_equal(arrays.multiplicity[:], np.ones(10, np.float32))


def test_a_selection_is_gathered_and_renumbered(
    ten_structure_dataset: Path, frames: list[Atoms]
) -> None:
    dataset = MoleculeDataset.open_existing_dataset_from_dir(ten_structure_dataset)
    chosen = np.array([1, 4, 5, 9])
    gathered = dataset.read_structures(chosen)
    expected = dataset.read_structures().take(chosen)
    for attribute in STRUCTURE_ARRAYS:
        np.testing.assert_array_equal(
            getattr(gathered, attribute), getattr(expected, attribute)
        )
    assert gathered.atom_counts.tolist() == [len(frames[i]) for i in chosen]


@pytest.mark.parametrize("structure_ids", [[2, 1], [1, 1], [-1], [10]])
def test_a_selection_must_be_sorted_unique_and_in_range(
    ten_structure_dataset: Path, structure_ids: list[int]
) -> None:
    dataset = MoleculeDataset.open_existing_dataset_from_dir(ten_structure_dataset)
    with pytest.raises(ValueError, match="structure_ids"):
        dataset.read_structures(np.array(structure_ids))


def test_a_directory_without_the_structure_arrays_is_refused(tmp_path: Path) -> None:
    zarr.open_group(tmp_path / "empty", mode="w")
    with pytest.raises(KeyError):
        MoleculeDataset.open_existing_dataset_from_dir(tmp_path / "empty")


def test_an_existing_dataset_opens_read_only(ten_structure_dataset: Path) -> None:
    dataset = MoleculeDataset.open_existing_dataset_from_dir(ten_structure_dataset)
    with pytest.raises(ValueError, match="read-only"):
        dataset.positions[0] = np.zeros(3)
    with pytest.raises(ValueError, match="read-only"):
        dataset.molecule_ids[0] = 99
    with pytest.raises(ValueError, match="read-only"):
        dataset.total_charge[0] = 5.0


def test_charges_follow_take_and_concatenate() -> None:
    structures = PackedStructures.from_frames(
        [(np.array([6]), np.zeros((1, 3))), (np.array([8, 1]), np.ones((2, 3)))]
    ).with_charges(np.array([0.0, -1.0]), np.array([1.0, 2.0]))
    assert structures.total_charge is not None
    assert structures.total_charge.dtype == np.float32
    taken = structures.take([1])
    assert taken.total_charge is not None and taken.multiplicity is not None
    assert (taken.total_charge.tolist(), taken.multiplicity.tolist()) == ([-1.0], [2.0])
    joined = PackedStructures.concatenate([structures, taken])
    assert joined.total_charge is not None
    assert joined.total_charge.tolist() == [0.0, -1.0, -1.0]
    with pytest.raises(ValueError, match="only some parts"):
        PackedStructures.concatenate([structures, PackedStructures.empty()])
    with pytest.raises(ValueError, match="total_charge has shape"):
        structures.with_charges(np.zeros(3), np.ones(2))
