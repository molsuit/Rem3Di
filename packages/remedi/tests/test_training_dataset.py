"""``TrainingMoleculeDataset`` reads samples from ``structures.atp``, one process at a time."""

from __future__ import annotations

import multiprocessing
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from torch.utils.data import DataLoader

from remedi.training.data.sample import Sample, yield_molecules_collate_fn
from remedi.training.data.training_dataset import (
    TrainingMoleculeDataset,
    atoms_getitem,
)
from testing import (
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


def test_samples_are_the_written_structures(
    ten_structure_dataset: Path, frames: list[Atoms]
) -> None:
    dataset = TrainingMoleculeDataset(ten_structure_dataset, atoms_getitem)
    assert len(dataset) == len(frames)
    assert dataset.atom_counts().tolist() == [len(frame) for frame in frames]
    for index, frame in enumerate(frames):
        sample = dataset[index]
        assert sample.atomic_positions is not None
        assert sample.atomic_numbers is not None
        assert sample.total_charge is not None
        assert sample.multiplicity is not None
        assert sample.atomic_positions.dtype == torch.float32
        assert sample.atomic_numbers.dtype == torch.uint8
        np.testing.assert_array_equal(
            sample.atomic_positions.numpy(), frame.positions.astype(np.float32)
        )
        np.testing.assert_array_equal(sample.atomic_numbers.numpy(), frame.numbers)
        assert float(sample.total_charge) == 0.0
        assert float(sample.multiplicity) == 1.0


def test_a_subset_is_indexed_in_its_own_order(
    ten_structure_dataset: Path, frames: list[Atoms]
) -> None:
    chosen = np.array([2, 5, 9])
    dataset = TrainingMoleculeDataset(
        ten_structure_dataset, atoms_getitem, structure_ids=chosen
    )
    assert len(dataset) == 3
    assert dataset.atom_counts().tolist() == [len(frames[i]) for i in chosen]
    for index, structure_id in enumerate(chosen):
        sample = dataset[index]
        assert sample.atomic_numbers is not None
        np.testing.assert_array_equal(
            sample.atomic_numbers.numpy(), frames[structure_id].numbers
        )


def test_loaders_with_and_without_workers_yield_identical_batches(
    ten_structure_dataset: Path,
) -> None:
    in_process = batches(
        TrainingMoleculeDataset(ten_structure_dataset, atoms_getitem), 0
    )
    # Two workers each open their own structure file and read through it.
    opened = TrainingMoleculeDataset(ten_structure_dataset, atoms_getitem).open()
    with_workers = batches(opened, 2)
    assert_same_batches(in_process, with_workers)
    # A pickled dataset (a spawned worker) reopens the file in its process.
    unpickled = pickle.loads(pickle.dumps(opened))
    assert unpickled._file is None
    assert_same_batches(in_process, batches(unpickled, 0))


#: The dataset a forked child inherits (not pickled) in the test below.
_INHERITED: list[TrainingMoleculeDataset] = []


def _file_process_seen_by_the_child(_: int) -> tuple[int, int | None]:
    (dataset,) = _INHERITED
    dataset._structure_file()
    return os.getpid(), dataset._file_process


def test_a_forked_process_opens_its_own_structure_file(
    ten_structure_dataset: Path,
) -> None:
    """A fork inherits the parent's open file; the dataset notices the new
    process and opens another before the first read."""
    dataset = TrainingMoleculeDataset(ten_structure_dataset, atoms_getitem).open()
    parent_file = dataset._structure_file()
    _INHERITED[:] = [dataset]
    try:
        with multiprocessing.get_context("fork").Pool(1) as pool:
            child, opened_by = pool.apply(_file_process_seen_by_the_child, (0,))
    finally:
        _INHERITED.clear()
    assert child != os.getpid() and opened_by == child
    assert dataset._structure_file() is parent_file


def test_worker_init_leaves_the_random_streams_alone() -> None:
    """torch seeds each worker; the init only limits threads and must not
    reseed numpy or ``random`` (which made every run's workers identical)."""
    import random

    import numpy as np

    from remedi.training.data.util import worker_init_fn

    np.random.seed(123)
    random.seed(123)
    expected_numpy = np.random.random()
    expected_random = random.random()
    np.random.seed(123)
    random.seed(123)
    worker_init_fn(3)
    assert np.random.random() == expected_numpy
    assert random.random() == expected_random
