"""zarr v3 sharding regression tests.

The dataset store used to be a one-file-per-chunk zarr v2 ``DirectoryStore``,
which exploded the on-disk file (inode) count and blew the 1M scratch quota.
zarr v3 sharding groups many chunks into a single shard file, and the
``ShardAlignedWriter`` buffers appended rows so each shard is written exactly
once. These tests pin both behaviours: tiny on-disk file count, and no
per-batch read-modify-write of the in-progress shard.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from zarr.storage import LoggingStore

import remedi.data_handling.dataset.molecule_dataset as md
from remedi.configuration.dataset_config import DatasetConfig
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset_creation.shard_aligned_writer import (
    ShardAlignedWriter,
)


def _count_files(root: Path) -> int:
    return sum(len(files) for _, _, files in os.walk(root))


def _append(writer: ShardAlignedWriter, n_mols: int, atoms_per: int) -> None:
    na = n_mols * atoms_per
    writer.append_batch(
        positions=np.zeros((na, 3), dtype="f4"),
        atomic_numbers=np.full(na, 6, dtype="u1"),
        batch_ptr_cumsum=np.arange(1, n_mols + 1) * atoms_per,
        molecule_ids=np.arange(n_mols),
        stereoisomer_ids=np.arange(n_mols),
        total_charge=np.zeros(n_mols, dtype="f4"),
        multiplicity=np.ones(n_mols, dtype="f4"),
        system_targets=None,
        system_masks=None,
        atom_targets=None,
        atom_masks=None,
    )


def test_sharded_store_keeps_file_count_tiny(tmp_path: Path) -> None:
    # Deliberately tiny chunks so the data spans MANY chunks. Under the old
    # one-file-per-chunk DirectoryStore this would be ~300+ files; sharding
    # must collapse them into a handful of shard files.
    cfg = DatasetConfig(
        atom_chunk=8,
        molecule_chunk=4,
        atom_chunks_per_shard=64,
        molecule_chunks_per_shard=64,
        contains_smiles=False,
    )
    root = tmp_path / "ds"
    ds = MoleculeDataset.create_empty_dataset(root, cfg)
    writer = ShardAlignedWriter(ds)

    n_mols = 500
    atoms_per = 5
    n_atoms = n_mols * atoms_per  # 2500 atoms -> ~312 atom-chunks of 8

    _append(writer, n_mols, atoms_per)
    writer.finalize()

    n_files = _count_files(root)
    # ~312 atom-chunks alone would be 300+ files under DirectoryStore.
    assert n_files < 60, f"sharding regressed: {n_files} files on disk"

    reopened = MoleculeDataset.open_existing_dataset_from_dir(root)
    assert reopened.N_structures == n_mols
    assert reopened.N_atoms == n_atoms
    np.testing.assert_array_equal(
        np.asarray(reopened.atomic_numbers[:n_atoms]),
        np.full(n_atoms, 6, dtype="u1"),
    )
    np.testing.assert_array_equal(
        np.asarray(reopened.multiplicity[:n_mols]),
        np.ones(n_mols, dtype="f4"),
    )


def test_no_per_batch_read_modify_write(tmp_path: Path, monkeypatch) -> None:
    """Appending many sub-shard batches must not rewrite the in-progress
    shard per batch.

    zarr v3 sharding does a full read-modify-write of a shard file for any
    write that doesn't cover whole shards. The shard-aligned writer must
    therefore produce: (1) ZERO store ``get`` calls during the append loop
    (no read-modify-write), and (2) a write count that scales with the shard
    count, not the (much larger) batch count.
    """
    created: dict[str, LoggingStore] = {}
    real_local_store = md.LocalStore

    def factory(path):
        s = LoggingStore(real_local_store(path))
        created["store"] = s
        return s

    monkeypatch.setattr(md, "LocalStore", factory)

    # atom_shard = 8 * 4 = 32 atoms; many tiny batches span ~31 shards.
    cfg = DatasetConfig(
        atom_chunk=8,
        atom_chunks_per_shard=4,
        molecule_chunk=64,
        molecule_chunks_per_shard=64,
        contains_smiles=False,
    )
    ds = MoleculeDataset.create_empty_dataset(tmp_path / "ds", cfg)
    writer = ShardAlignedWriter(ds)
    store = created["store"]

    n_batches, mols_per_batch, atoms_per = 20, 10, 5
    total_atoms = n_batches * mols_per_batch * atoms_per  # 1000
    atom_shard = cfg.atom_chunk * cfg.atom_chunks_per_shard  # 32

    before = dict(store.counter)
    for _ in range(n_batches):
        _append(writer, mols_per_batch, atoms_per)
    after_loop = dict(store.counter)
    writer.finalize()

    def delta(a, b, k):
        return b.get(k, 0) - a.get(k, 0)

    gets_during_append = delta(before, after_loop, "get")
    sets_during_append = delta(before, after_loop, "set")

    # (1) the core invariant: no read-modify-write while appending.
    assert gets_during_append == 0, (
        f"read-modify-write during append: {gets_during_append} store gets "
        "(must be 0 — the shard-aligned buffer was bypassed)"
    )
    # (2) writes scale with shards, not batches: with one set per shard for
    # positions + atomic_numbers (+ metadata), the count is far below what a
    # per-batch write path would produce (which would be O(n_batches) writes
    # into the in-progress shard *per array*).
    n_atom_shards = total_atoms // atom_shard
    assert sets_during_append <= 4 * n_atom_shards + 16, (
        f"too many writes ({sets_during_append}) for {n_atom_shards} shards "
        f"over {n_batches} batches — looks per-batch, not per-shard"
    )

    reopened = MoleculeDataset.open_existing_dataset_from_dir(tmp_path / "ds")
    assert reopened.N_atoms == total_atoms
    assert reopened.N_structures == n_batches * mols_per_batch


def test_all_buffered_until_shard_complete(tmp_path: Path, monkeypatch) -> None:
    """If every appended batch fits within a single (not-yet-complete)
    shard, nothing is written to the store until finalize — proving writes
    happen per shard, never per batch."""
    created: dict[str, LoggingStore] = {}
    real_local_store = md.LocalStore

    def factory(path):
        s = LoggingStore(real_local_store(path))
        created["store"] = s
        return s

    monkeypatch.setattr(md, "LocalStore", factory)

    # atom_shard = 16 * 64 = 1024; total atoms below one shard.
    cfg = DatasetConfig(
        atom_chunk=16,
        atom_chunks_per_shard=64,
        molecule_chunk=64,
        molecule_chunks_per_shard=64,
        contains_smiles=False,
    )
    ds = MoleculeDataset.create_empty_dataset(tmp_path / "ds", cfg)
    writer = ShardAlignedWriter(ds)
    store = created["store"]

    before = dict(store.counter)
    for _ in range(40):  # 40 * 5 * 5 = 1000 atoms < 1024 (one shard)
        _append(writer, 5, 5)
    after_loop = dict(store.counter)

    # Nothing flushed yet: no data writes happened during the append loop.
    assert after_loop.get("set", 0) == before.get(
        "set", 0
    ), "data was written before a shard completed — buffering bypassed"
    assert after_loop.get("get", 0) == before.get("get", 0)

    writer.finalize()  # single partial-shard write here
    reopened = MoleculeDataset.open_existing_dataset_from_dir(tmp_path / "ds")
    assert reopened.N_atoms == 1000
    assert reopened.N_structures == 200
    np.testing.assert_array_equal(
        np.asarray(reopened.atomic_numbers[:1000]),
        np.full(1000, 6, dtype="u1"),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
