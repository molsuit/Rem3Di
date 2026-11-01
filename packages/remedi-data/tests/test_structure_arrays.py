"""The zarr structure arrays as ``remedi_data`` exposes them to training code."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import zarr
from ase import Atoms

from remedi_data.dataset.molecule_dataset import MoleculeDataset, StructureArrays
from remedi_data.testing import (
    ACHIRAL_TEN_SMILES,
    embed_frame,
    write_ten_structure_dataset,
)


@pytest.fixture
def frames() -> list[Atoms]:
    return [embed_frame(smiles) for smiles in ACHIRAL_TEN_SMILES]


@pytest.fixture
def ten_structure_dataset(tmp_path: Path, frames: list[Atoms]) -> Path:
    return write_ten_structure_dataset(tmp_path, frames)


def test_the_structure_array_names_are_the_zarr_array_names(
    ten_structure_dataset: Path,
) -> None:
    group = zarr.open_group(ten_structure_dataset, mode="r")
    for field in fields(StructureArrays):
        assert field.name in group, f"{field.name} is not an array in the zarr"


@pytest.mark.parametrize("in_memory", [True, False])
def test_structure_arrays_hold_the_written_structures(
    ten_structure_dataset: Path, frames: list[Atoms], in_memory: bool
) -> None:
    arrays = MoleculeDataset.open_structure_arrays(
        ten_structure_dataset, in_memory=in_memory
    )
    for field in fields(StructureArrays):
        value = getattr(arrays, field.name)
        assert isinstance(value, np.ndarray) is in_memory

    molecule_ptr = np.asarray(arrays.molecule_ptr[:])
    assert molecule_ptr.tolist() == np.cumsum([0] + [len(f) for f in frames]).tolist()
    for index, frame in enumerate(frames):
        start, end = molecule_ptr[index], molecule_ptr[index + 1]
        np.testing.assert_array_equal(
            arrays.positions[start:end], frame.positions.astype(np.float32)
        )
        np.testing.assert_array_equal(
            arrays.atomic_numbers[start:end], frame.numbers.astype(np.uint8)
        )
    np.testing.assert_array_equal(arrays.total_charge[:], np.zeros(10, np.float32))
    np.testing.assert_array_equal(arrays.multiplicity[:], np.ones(10, np.float32))


def test_structure_arrays_are_opened_read_only(ten_structure_dataset: Path) -> None:
    arrays = MoleculeDataset.open_structure_arrays(ten_structure_dataset)
    with pytest.raises(Exception):  # noqa: B017 - zarr's read-only error type varies by store
        arrays.total_charge[0] = 5.0


def test_a_directory_without_the_structure_arrays_is_refused(tmp_path: Path) -> None:
    zarr.open_group(tmp_path / "empty", mode="w")
    with pytest.raises(KeyError, match="molecule_ptr"):
        MoleculeDataset.open_structure_arrays(tmp_path / "empty")
