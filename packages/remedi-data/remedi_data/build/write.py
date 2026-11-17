"""``write_dataset``: structures + table -> a zarr with the table files beside it.

The one writer every structure dataset goes through (§4.1): the preparer of a
source that ships structures calls it directly, and ``dataset_build`` calls it
after generating conformers for a SMILES bundle. It validates the table and
every structure (invariants 1-10), writes the zarr
(:meth:`MoleculeDataset.write`, chunked and sharded as ``layout`` says) into a
staging directory, checks the zarr against
the table (invariant 8), and only then moves the directory into place and
writes ``dataset.yaml``, ``table.parquet`` and ``provenance.yaml`` beside it.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from remedi_data.dataset.config import DatasetConfig
from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.dataset.structures import Structures
from remedi_data.format import (
    BundleProvenance,
    BundleValidationError,
    DatasetSpec,
    normalize_table,
    structure_problems,
    validate_table,
    write_table_files,
)
from remedi_data.format.verify import structures_sha256, zarr_problems

logger = logging.getLogger(__name__)


def _write_zarr(
    path: Path,
    table: pd.DataFrame,
    structures: Structures,
    layout: DatasetConfig,
) -> None:
    MoleculeDataset.write(
        path,
        structures.with_charges(
            table["total_charge"].to_numpy(dtype=np.float64),
            table["multiplicity"].to_numpy(dtype=np.float64),
        ),
        structure_ids=table["structure_id"].to_numpy(dtype=np.int64),
        molecule_ids=table["molecule_id"].to_numpy(dtype=np.int64),
        stereoisomer_ids=table["stereoisomer_id"].to_numpy(dtype=np.int64),
        config=layout,
    )


def write_dataset(
    spec: DatasetSpec,
    table: pd.DataFrame,
    structures: Structures,
    provenance: BundleProvenance,
    directory: Path,
    *,
    layout: DatasetConfig | None = None,
    workers: int = 1,
) -> BundleProvenance:
    """Validate and write one dataset; returns the provenance as written.

    The geometry limits checked (invariant 10) are ``provenance.geometry_limits``.
    ``workers`` processes check invariants 9 and 10.
    An existing dataset at ``directory`` is replaced only after the new one has
    been written and checked in full (see :func:`_move_into_place`).

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
    problems += structure_problems(
        spec, table, structures, provenance.geometry_limits, workers=workers
    )
    if problems:
        raise BundleValidationError(directory, problems)

    staging = directory.with_name(f".{directory.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)  # left by an earlier run that died mid-write
    try:
        _write_zarr(staging, table, structures, layout or DatasetConfig())
        problems = zarr_problems(staging, table)
        if problems:
            raise BundleValidationError(directory, problems)
        dataset = MoleculeDataset.open_existing_dataset_from_dir(staging)
        structure_hash = structures_sha256(dataset)
        written = write_table_files(
            staging, spec, table, provenance, structures_sha256=structure_hash
        )
        _move_into_place(staging, directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return written


def _move_into_place(staging: Path, directory: Path) -> None:
    """Replace ``directory`` by ``staging`` without a moment in which neither exists.

    The old dataset is renamed aside first, so nothing it holds (an NFS
    ``.nfs*`` file a reader keeps open, an entry without write permission) can
    stop the new one from taking its name. If the new one cannot be moved in,
    the old one is moved back. Removing the set-aside copy comes last and only
    costs disk space when it fails, so that failure is logged, not raised.

    Raises:
        OSError: if either rename fails; ``directory`` then holds what it held.
    """
    if not directory.exists():
        staging.rename(directory)
        return
    set_aside = directory.with_name(f".{directory.name}.old-{uuid.uuid4().hex}")
    directory.rename(set_aside)
    try:
        staging.rename(directory)
    except OSError:
        set_aside.rename(directory)
        raise
    try:
        shutil.rmtree(set_aside)
    except OSError as error:
        logger.warning(
            "the replaced dataset %s could not be removed (%s); delete it by hand",
            set_aside,
            error,
        )
