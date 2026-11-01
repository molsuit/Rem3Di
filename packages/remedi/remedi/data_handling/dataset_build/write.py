"""``write_dataset``: structures + table -> a zarr with the table files beside it.

The one writer every structure dataset goes through (§10.1): the preparer of a
source that ships structures calls it directly, and ``dataset_build`` calls it
after generating conformers for a SMILES bundle. It validates the table and
every structure (invariants 1-10), writes the zarr through
:class:`ShardAlignedWriter` into a staging directory, checks the zarr against
the table (invariant 8), and only then moves the directory into place and
writes ``dataset.yaml``, ``table.parquet`` and ``provenance.yaml`` beside it.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field

from remedi.configuration.dataset_config import DatasetConfig
from remedi.data_handling.bundle import (
    BundleProvenance,
    BundleValidationError,
    DatasetSpec,
    normalize_table,
    structure_problems,
    validate_table,
    write_table_files,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset.tasks import Split, split_codes_from_names
from remedi.data_handling.dataset_creation.shard_aligned_writer import (
    ShardAlignedWriter,
)


class ZarrLayout(BaseModel):
    """Chunk and shard geometry of the written zarr; defaults match ``DatasetConfig``."""

    model_config = ConfigDict(extra="forbid")

    atom_chunk: int = Field(default=8192, ge=1)
    molecule_chunk: int = Field(default=4096, ge=1)
    atom_chunks_per_shard: int = Field(default=64, ge=1)
    molecule_chunks_per_shard: int = Field(default=256, ge=1)
    #: Structures handed to the writer per append.
    batch_size: int = Field(default=4096, ge=1)


def structures_sha256(dataset: MoleculeDataset) -> str:
    """sha256 over the zarr's ``atomic_numbers``, ``positions`` and ``ptr``.

    The data identity a descriptor cache keys on: it moves whenever any
    structure, or the set of structures, changes.
    """
    digest = hashlib.sha256()
    for name, array in (
        ("atomic_numbers", dataset.atomic_numbers),
        ("positions", dataset.positions),
        ("molecule_ptr", dataset.ptr),
    ):
        values = np.ascontiguousarray(np.asarray(array[:]))
        digest.update(f"{name}:{values.dtype.str}:{values.shape}\n".encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _labels_and_masks(
    table: pd.DataFrame, spec: DatasetSpec
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not spec.labels:
        return None, None
    values = table[spec.label_names()].to_numpy(dtype=np.float64)
    masks = (~np.isnan(values)).astype(np.uint8)
    return np.where(masks.astype(bool), values, 0.0), masks


def _split_codes(table: pd.DataFrame, spec: DatasetSpec) -> np.ndarray:
    if spec.evaluation is None:
        return np.full(len(table), Split.unassigned.value, dtype=np.uint8)
    return split_codes_from_names(
        table[spec.evaluation.default_split].astype(str).tolist()
    )


def _write_zarr(
    path: Path,
    spec: DatasetSpec,
    table: pd.DataFrame,
    structures: Sequence[Atoms],
    layout: ZarrLayout,
) -> None:
    dataset = MoleculeDataset.create_empty_dataset(
        path,
        DatasetConfig(
            atom_chunk=layout.atom_chunk,
            molecule_chunk=layout.molecule_chunk,
            atom_chunks_per_shard=layout.atom_chunks_per_shard,
            molecule_chunks_per_shard=layout.molecule_chunks_per_shard,
            contains_smiles=False,
            tasks=spec.task_set() if spec.labels else None,
        ),
    )
    writer = ShardAlignedWriter(dataset)
    targets, masks = _labels_and_masks(table, spec)
    split_codes = _split_codes(table, spec)
    structure_ids = table["structure_id"].to_numpy(dtype=np.int64)
    molecule_ids = table["molecule_id"].to_numpy(dtype=np.int64)
    stereoisomer_ids = table["stereoisomer_id"].to_numpy(dtype=np.int64)
    total_charge = table["total_charge"].to_numpy(dtype=np.float64)
    multiplicity = table["multiplicity"].to_numpy(dtype=np.float64)
    try:
        for start in range(0, len(table), layout.batch_size):
            stop = min(start + layout.batch_size, len(table))
            batch = structures[start:stop]
            atom_counts = np.array([len(atoms) for atoms in batch], dtype=np.int64)
            writer.append_batch(
                np.concatenate([atoms.get_positions() for atoms in batch], axis=0),
                np.concatenate([atoms.get_atomic_numbers() for atoms in batch]),
                np.cumsum(atom_counts),
                molecule_ids[start:stop],
                stereoisomer_ids[start:stop],
                total_charge[start:stop],
                multiplicity[start:stop],
                None if targets is None else targets[start:stop],
                None if masks is None else masks[start:stop],
                None,
                None,
                split_codes[start:stop],
                structure_ids=structure_ids[start:stop],
            )
        writer.finalize()
    finally:
        dataset.close()


def zarr_problems(path: Path, spec: DatasetSpec, table: pd.DataFrame) -> list[str]:
    """Invariant 8: the zarr at ``path`` against the table, row for row.

    Structure count, the three id arrays, the default split's codes and the
    label values must all equal what the table says.
    """
    dataset = MoleculeDataset.open_existing_dataset_from_dir(path)
    try:
        rows = len(table)
        if dataset.N_structures != rows:
            return [
                f"8: the zarr holds {dataset.N_structures} structures for {rows} rows"
            ]
        problems: list[str] = []
        expected_ids = {
            "ids/structure_id": (dataset.structure_ids, table["structure_id"]),
            "ids/molecule_id": (dataset.molecule_ids, table["molecule_id"]),
            "ids/stereoisomer_id": (dataset.isomer_ids, table["stereoisomer_id"]),
        }
        for name, (array, column) in expected_ids.items():
            if array is None or not np.array_equal(
                np.asarray(array[:rows], dtype=np.int64),
                column.to_numpy(dtype=np.int64),
            ):
                problems.append(f"8: {name} does not match the table")
        # A corpus without labels has no task arrays, so no tasks/split either.
        if spec.evaluation is not None and (
            dataset.split is None
            or not np.array_equal(
                np.asarray(dataset.split[:rows], dtype=np.uint8),
                _split_codes(table, spec),
            )
        ):
            problems.append("8: tasks/split does not match the table's default split")
        targets, masks = _labels_and_masks(table, spec)
        if targets is not None and masks is not None:
            stored_masks = np.asarray(dataset.mask_system[:rows]).astype(np.uint8)
            stored_targets = np.asarray(dataset.targets_system[:rows], dtype=np.float64)
            if not np.array_equal(stored_masks, masks) or not np.allclose(
                np.where(masks.astype(bool), stored_targets, 0.0), targets
            ):
                problems.append("8: the zarr's labels do not match the table")
        return problems
    finally:
        dataset.close()


def write_dataset(
    spec: DatasetSpec,
    table: pd.DataFrame,
    structures: Sequence[Atoms],
    provenance: BundleProvenance,
    directory: Path,
    *,
    layout: ZarrLayout | None = None,
) -> BundleProvenance:
    """Validate and write one dataset; returns the provenance as written.

    The geometry limits checked (invariant 10) are ``provenance.geometry_limits``.
    An existing dataset at ``directory`` is replaced only after the new one has
    been written and checked in full.

    Raises:
        BundleValidationError: if the spec declares no structures, or any
            invariant fails on the table, the structures or the written zarr.
            Nothing is left at ``directory`` in that case beyond what was there.
    """
    directory = Path(directory)
    if not spec.has_structures:
        raise BundleValidationError(
            directory, ["a dataset must declare geometry_origin"]
        )
    table = normalize_table(table, spec)
    problems = validate_table(spec, table)
    problems += structure_problems(spec, table, structures, provenance.geometry_limits)
    if problems:
        raise BundleValidationError(directory, problems)

    staging = directory.with_name(f".{directory.name}.partial")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        _write_zarr(staging, spec, table, structures, layout or ZarrLayout())
        problems = zarr_problems(staging, spec, table)
        if problems:
            raise BundleValidationError(directory, problems)
        dataset = MoleculeDataset.open_existing_dataset_from_dir(staging)
        try:
            structure_hash = structures_sha256(dataset)
        finally:
            dataset.close()
        written = write_table_files(
            staging, spec, table, provenance, structures_sha256=structure_hash
        )
        shutil.rmtree(directory, ignore_errors=True)
        staging.rename(directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return written
