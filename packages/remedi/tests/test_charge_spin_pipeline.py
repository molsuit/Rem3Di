"""Charge and multiplicity travel from the dataset to the training batch.

``TrainingMoleculeDataset`` exposes each structure's ``total_charge`` and
``multiplicity`` per ``Sample`` via ``atoms_getitem``, and
``yield_molecules_collate_fn`` batches them per system.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from remedi.training.data.sample import Sample, yield_molecules_collate_fn
from remedi.training.data.training_dataset import (
    TrainingMoleculeDataset,
    atoms_getitem,
)
from testing import write_small_dataset


def test_atoms_getitem_returns_charge_and_spin_per_sample(tmp_path: Path) -> None:
    # A cation (charge +1, singlet) and a methyl radical (neutral, doublet).
    dataset = write_small_dataset(
        tmp_path,
        dataset_id="charged",
        labels=[],
        metrics=None,
        targets=None,
        smiles=["C[NH3+]", "[CH3]"],
    )
    train_ds = TrainingMoleculeDataset(dataset, atoms_getitem)
    assert len(train_ds) == 2
    cation, radical = train_ds[0], train_ds[1]
    torch.testing.assert_close(cation.total_charge, torch.tensor(1.0))
    torch.testing.assert_close(cation.multiplicity, torch.tensor(1.0))
    torch.testing.assert_close(radical.total_charge, torch.tensor(0.0))
    torch.testing.assert_close(radical.multiplicity, torch.tensor(2.0))


def test_yield_molecules_collate_batches_charge_and_spin() -> None:
    s1 = Sample(
        atomic_positions=torch.zeros((2, 3)),
        atomic_numbers=torch.tensor([1, 1], dtype=torch.uint8),
        total_charge=torch.tensor(1.0),
        multiplicity=torch.tensor(2.0),
    )
    s2 = Sample(
        atomic_positions=torch.zeros((3, 3)),
        atomic_numbers=torch.tensor([6, 1, 1], dtype=torch.uint8),
        total_charge=torch.tensor(-1.0),
        multiplicity=torch.tensor(0.5),
    )

    batched = yield_molecules_collate_fn([s1, s2])

    assert batched.atomic_positions.shape == (5, 3)
    assert batched.atomic_numbers.shape == (5,)
    torch.testing.assert_close(
        batched.system_index, torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    )
    torch.testing.assert_close(batched.total_charge, torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(batched.multiplicity, torch.tensor([2.0, 0.5]))


def test_yield_molecules_collate_requires_charge_and_spin() -> None:
    s = Sample(
        atomic_positions=torch.zeros((1, 3)),
        atomic_numbers=torch.tensor([1], dtype=torch.uint8),
    )
    with pytest.raises((TypeError, ValueError, AttributeError)):
        yield_molecules_collate_fn([s])
