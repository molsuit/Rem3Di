"""The zarr layout ``MoleculeDataset.write`` produces.

The dataset store used to be a one-file-per-chunk zarr v2 ``DirectoryStore``,
which exploded the on-disk file (inode) count and blew the 1M scratch quota.
zarr v3 sharding groups many chunks into a single shard file, and the writer
hands each array to zarr in one piece, so each shard is written exactly once.
These tests pin the chunk and shard shapes, the tiny file count, the absence of
a read-modify-write, and the structure hash of a fixed dataset.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

import remedi_data.dataset.molecule_dataset as molecule_dataset_module
from remedi_data.build import structures_sha256
from remedi_data.dataset.config import DatasetConfig
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.dataset.packed_structures import PackedStructures
from remedi_data.testing import write_synthetic_dataset


def _carbon_structures(n_molecules: int, atoms_per_molecule: int) -> PackedStructures:
    n_atoms = n_molecules * atoms_per_molecule
    return PackedStructures.from_arrays(
        np.zeros((n_atoms, 3)),
        np.full(n_atoms, 6),
        np.arange(n_molecules + 1) * atoms_per_molecule,
        np.zeros(n_molecules),
        np.ones(n_molecules),
    )


def _write(
    path: Path,
    structures: PackedStructures,
    config: DatasetConfig,
    molecule_ids: np.ndarray | None = None,
) -> MoleculeDataset:
    count = len(structures)
    return MoleculeDataset.write(
        path,
        structures,
        structure_ids=np.arange(count),
        molecule_ids=np.arange(count) if molecule_ids is None else molecule_ids,
        stereoisomer_ids=np.arange(count),
        config=config,
    )


SMALL = DatasetConfig(
    atom_chunk=8, molecule_chunk=4, atom_chunks_per_shard=4, molecule_chunks_per_shard=4
)


def test_chunks_and_shards_follow_the_config(tmp_path: Path) -> None:
    _write(tmp_path / "ds", _carbon_structures(50, 3), SMALL)
    group = zarr.open_group(tmp_path / "ds", mode="r")
    expected = {
        "positions": ((8, 3), (32, 3), (150, 3)),
        "atomic_numbers": ((8,), (32,), (150,)),
        "molecule_ptr": ((4,), (16,), (51,)),
        "total_charge": ((4,), (16,), (50,)),
        "multiplicity": ((4,), (16,), (50,)),
        "ids/structure_id": ((4,), (16,), (50,)),
        "ids/molecule_id": ((4,), (16,), (50,)),
        "ids/stereoisomer_id": ((4,), (16,), (50,)),
    }
    for name, (chunks, shards, shape) in expected.items():
        array = group[name]
        assert isinstance(array, zarr.Array)
        assert (array.chunks, array.shards, array.shape) == (chunks, shards, shape), (
            name
        )


def test_sharded_store_keeps_file_count_tiny(tmp_path: Path) -> None:
    # Deliberately tiny chunks so the data spans MANY chunks (~312 atom chunks
    # of 8, 300+ files under a one-file-per-chunk store).
    config = DatasetConfig(
        atom_chunk=8,
        molecule_chunk=4,
        atom_chunks_per_shard=64,
        molecule_chunks_per_shard=64,
    )
    root = tmp_path / "ds"
    dataset = _write(root, _carbon_structures(500, 5), config)
    n_files = sum(len(files) for _, _, files in os.walk(root))
    assert n_files < 60, f"sharding regressed: {n_files} files on disk"
    assert dataset.N_structures == 500
    assert dataset.N_atoms == 2500
    np.testing.assert_array_equal(
        np.asarray(dataset.atomic_numbers[:]), np.full(2500, 6, dtype="u1")
    )


def test_each_shard_is_written_once_without_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """zarr v3 does a read-modify-write of a shard for any partial write; the
    writer covers whole shards, so a data key is set once per shard and read
    at most once per array (the trailing, partially filled shard)."""
    data_gets: Counter[str] = Counter()
    data_sets: Counter[str] = Counter()

    class CountingStore(LocalStore):
        async def get(self, key, *args, **kwargs):
            if "/c/" in key:
                data_gets[key.split("/c/")[0]] += 1
            return await super().get(key, *args, **kwargs)

        async def set(self, key, value, *args, **kwargs):
            if "/c/" in key:
                data_sets[key] += 1
            return await super().set(key, value, *args, **kwargs)

    monkeypatch.setattr(molecule_dataset_module, "LocalStore", CountingStore)
    config = DatasetConfig(
        atom_chunk=8,
        atom_chunks_per_shard=4,
        molecule_chunk=64,
        molecule_chunks_per_shard=64,
    )
    structures = _carbon_structures(200, 5)
    _write(tmp_path / "ds", structures, config)
    assert max(data_gets.values()) <= 1
    assert set(data_sets.values()) == {1}
    atom_shards = -(-1000 // 32)
    assert sum("atomic_numbers" in key for key in data_sets) == atom_shards


def test_an_empty_dataset_can_be_written(tmp_path: Path) -> None:
    dataset = _write(tmp_path / "ds", _carbon_structures(0, 3), SMALL)
    assert (dataset.N_structures, dataset.N_atoms) == (0, 0)
    assert len(dataset.read_structures()) == 0


def test_the_ids_are_written_verbatim(tmp_path: Path) -> None:
    dataset = MoleculeDataset.write(
        tmp_path / "ds",
        _carbon_structures(5, 3),
        structure_ids=np.array([3, 1, 4, 1, 5]),
        molecule_ids=np.array([0, 7, 7, 8, 8]),
        stereoisomer_ids=np.arange(5),
        config=SMALL,
    )
    np.testing.assert_array_equal(dataset.structure_ids[:], [3, 1, 4, 1, 5])
    # Ids stay as the bundle assigned them, so a dropped molecule leaves a gap.
    assert dataset.N_molecules == 3


def test_a_wrong_length_id_array_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"structure_ids has shape \(4,\)"):
        MoleculeDataset.write(
            tmp_path / "ds",
            _carbon_structures(5, 3),
            structure_ids=np.arange(4),
            molecule_ids=np.arange(5),
            stereoisomer_ids=np.arange(5),
            config=SMALL,
        )


def test_structures_without_charges_are_refused(tmp_path: Path) -> None:
    structures = PackedStructures.from_arrays(
        np.zeros((3, 3)), np.full(3, 6), np.array([0, 3])
    )
    with pytest.raises(ValueError, match="charges"):
        _write(tmp_path / "ds", structures, SMALL)


def test_the_structure_hash_of_a_fixed_dataset_is_pinned(tmp_path: Path) -> None:
    """The value the shard-aligned writer produced before ``create_array``
    replaced it: the switch moved no byte the hash covers."""
    write_synthetic_dataset(
        tmp_path / "synthetic",
        ["CCO", "c1ccccc1", "CCN", "CCCl"] * 10,
        structures_per_molecule=3,
    )
    dataset = MoleculeDataset.open_existing_dataset_from_dir(tmp_path / "synthetic")
    assert structures_sha256(dataset) == (
        "f3a84caadf9163df410cb1adc07417a44413d4a744660e93abe8073ff1198715"
    )
