"""Round-trip tests for the per-structure literature `split` column.

Splits are materialized into the dataset at ingest (Split enum -> uint8 zarr
column under the `tasks` group) and threaded through the ShardAlignedWriter
parallel to `targets_system`. These tests pin: (1) the column only exists when
the dataset has tasks, (2) supplied splits survive a write/finalize/reopen
cycle across shard boundaries, (3) an absent split defaults to
Split.unassigned so the molecule axis stays aligned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from remedi.configuration.dataset_config import DatasetConfig
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset.tasks import (
    Split,
    TaskConfig,
    TaskScope,
    TaskSet,
    TaskType,
)
from remedi.data_handling.dataset_creation.shard_aligned_writer import (
    ShardAlignedWriter,
)


def _task_config() -> DatasetConfig:
    return DatasetConfig(
        atom_chunk=8,
        molecule_chunk=4,
        atom_chunks_per_shard=4,
        molecule_chunks_per_shard=4,
        contains_smiles=False,
        tasks=TaskSet.from_list(
            [
                TaskConfig(
                    name="t0",
                    task_type=TaskType.regression,
                    scope=TaskScope.system,
                )
            ]
        ),
    )


def _append(writer: ShardAlignedWriter, n_mols: int, atoms_per: int, split=None):
    na = n_mols * atoms_per
    writer.append_batch(
        positions=np.zeros((na, 3), dtype="f4"),
        atomic_numbers=np.full(na, 6, dtype="u1"),
        batch_ptr_cumsum=np.arange(1, n_mols + 1) * atoms_per,
        molecule_ids=np.arange(n_mols),
        stereoisomer_ids=np.arange(n_mols),
        total_charge=np.zeros(n_mols, dtype="f4"),
        multiplicity=np.ones(n_mols, dtype="f4"),
        system_targets=np.zeros((n_mols, 1), dtype="f4"),
        system_masks=np.ones((n_mols, 1), dtype="u1"),
        atom_targets=None,
        atom_masks=None,
        split=split,
    )


def test_split_array_absent_without_tasks(tmp_path: Path) -> None:
    cfg = DatasetConfig(
        atom_chunk=8,
        molecule_chunk=4,
        contains_smiles=False,
    )
    ds = MoleculeDataset.create_empty_dataset(tmp_path / "ds", cfg)
    assert ds.split is None
    writer = ShardAlignedWriter(ds)
    # No tasks -> writer.append_batch with split must stay a no-op for split.
    writer.append_batch(
        positions=np.zeros((10, 3), dtype="f4"),
        atomic_numbers=np.full(10, 6, dtype="u1"),
        batch_ptr_cumsum=np.arange(1, 3) * 5,
        molecule_ids=np.arange(2),
        stereoisomer_ids=np.arange(2),
        total_charge=np.zeros(2, dtype="f4"),
        multiplicity=np.ones(2, dtype="f4"),
        system_targets=None,
        system_masks=None,
        atom_targets=None,
        atom_masks=None,
    )
    writer.finalize()
    assert MoleculeDataset.open_existing_dataset_from_dir(tmp_path / "ds").split is None


def test_split_round_trips_through_writer(tmp_path: Path) -> None:
    cfg = _task_config()
    root = tmp_path / "ds"
    ds = MoleculeDataset.create_empty_dataset(root, cfg)
    assert ds.split is not None
    writer = ShardAlignedWriter(ds)

    # Two batches spanning the molecule shard grid (shard = 4 * 4 = 16 mols).
    rng = np.random.default_rng(0)
    split_a = rng.integers(0, 3, size=20, dtype="u1")
    split_b = rng.integers(0, 3, size=17, dtype="u1")
    _append(writer, 20, 3, split=split_a)
    _append(writer, 17, 3, split=split_b)
    writer.finalize()

    reopened = MoleculeDataset.open_existing_dataset_from_dir(root)
    expected = np.concatenate([split_a, split_b])
    np.testing.assert_array_equal(
        np.asarray(reopened.split[: reopened.N_structures]), expected
    )
    assert reopened.N_structures == 37


def test_split_defaults_unassigned_when_not_supplied(tmp_path: Path) -> None:
    cfg = _task_config()
    root = tmp_path / "ds"
    ds = MoleculeDataset.create_empty_dataset(root, cfg)
    writer = ShardAlignedWriter(ds)
    _append(writer, 9, 4, split=None)
    writer.finalize()

    reopened = MoleculeDataset.open_existing_dataset_from_dir(root)
    np.testing.assert_array_equal(
        np.asarray(reopened.split[:9]),
        np.full(9, Split.unassigned.value, dtype="u1"),
    )
