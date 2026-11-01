"""``TrainingMoleculeDataset`` reads the zarr through ``MoleculeDataset.open_structure_arrays``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from torch.utils.data import DataLoader

from remedi.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    atoms_getitem,
)
from remedi.data_handling.sample import Sample, yield_molecules_collate_fn
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


def batches(dataset: TrainingMoleculeDataset, num_workers: int) -> list[Sample]:
    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=yield_molecules_collate_fn,
    )
    return list(loader)


def assert_same_batches(left: list[Sample], right: list[Sample]) -> None:
    assert len(left) == len(right)
    for left_batch, right_batch in zip(left, right, strict=True):
        for name in (
            "atomic_positions",
            "atomic_numbers",
            "system_index",
            "total_charge",
            "multiplicity",
        ):
            torch.testing.assert_close(
                getattr(left_batch, name), getattr(right_batch, name), rtol=0, atol=0
            )


@pytest.mark.parametrize("in_memory", [True, False])
def test_samples_are_the_written_structures(
    ten_structure_dataset: Path, frames: list[Atoms], in_memory: bool
) -> None:
    dataset = TrainingMoleculeDataset(
        ten_structure_dataset, atoms_getitem, in_memory=in_memory
    )
    assert len(dataset) == len(frames)
    for index, frame in enumerate(frames):
        sample = dataset[index]
        assert sample.atomic_positions is not None
        assert sample.atomic_numbers is not None
        assert sample.total_charge is not None
        assert sample.multiplicity is not None
        np.testing.assert_array_equal(
            sample.atomic_positions.numpy(), frame.positions.astype(np.float32)
        )
        np.testing.assert_array_equal(sample.atomic_numbers.numpy(), frame.numbers)
        assert float(sample.total_charge) == 0.0
        assert float(sample.multiplicity) == 1.0


def test_lazy_in_memory_and_worker_loaders_yield_identical_batches(
    ten_structure_dataset: Path,
) -> None:
    in_memory = batches(
        TrainingMoleculeDataset(ten_structure_dataset, atoms_getitem, in_memory=True), 0
    )
    lazy = batches(TrainingMoleculeDataset(ten_structure_dataset, atoms_getitem), 0)
    # Workers receive a pickled dataset, which drops its arrays and reopens them.
    with_workers = batches(
        TrainingMoleculeDataset(ten_structure_dataset, atoms_getitem), 2
    )
    assert_same_batches(in_memory, lazy)
    assert_same_batches(in_memory, with_workers)
