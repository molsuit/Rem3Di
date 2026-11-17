"""``write_dataset``: table + structures -> a dataset directory.

The one writer every structure dataset goes through (§4.1): the preparer of a
source that ships structures calls it directly, and ``dataset_build`` calls it
after generating conformers for a SMILES bundle. It validates the table, then
streams the structures part by part (a shard of a corpus, or all frames of a
benchmark at once): each part is checked (invariants 9 and 10) and appended to
``structures.atp`` and dropped, so memory holds the table and one part. Into a
staging directory it writes ``structures.atp`` (flushed once), then
``dataset.yaml``, ``table.parquet`` (one row group per part) and, last,
``provenance.yaml`` with the hashes, and only then moves the directory into
place.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from remedi_data.dataset.dataset import structures_sha256
from remedi_data.dataset.structure_file import StructureFile, write_structures
from remedi_data.dataset.structures import Structures
from remedi_data.format import (
    STRUCTURES_FILENAME,
    BundleProvenance,
    BundleValidationError,
    DatasetSpec,
    StructureChecker,
    normalize_table,
    validate_table,
    write_table_files,
)

logger = logging.getLogger(__name__)


def write_dataset(
    spec: DatasetSpec,
    table: pd.DataFrame,
    structure_parts: Iterable[Structures],
    provenance: BundleProvenance,
    directory: Path,
    *,
    workers: int = 1,
) -> BundleProvenance:
    """Validate and write one dataset; returns the provenance as written.

    ``structure_parts`` are the frames of the table's rows in order, in any
    number of parts (charges and multiplicities come from the table; a part's
    own are ignored). They are consumed once, one at a time. The geometry
    limits checked (invariant 10) are ``provenance.geometry_limits``;
    ``workers`` processes check invariants 9 and 10. An existing dataset at
    ``directory`` is replaced only after the new one has been written and
    checked in full (see :func:`_move_into_place`).

    Raises:
        BundleValidationError: if the spec declares no structures, or any
            invariant fails on the table or the structures (all problems are
            listed). Nothing is left at ``directory`` in that case beyond what
            was there.
    """
    directory = Path(directory)
    if not spec.has_structures:
        raise BundleValidationError(
            directory, ["a dataset must declare geometry_origin"]
        )
    table = normalize_table(table, spec)
    table_problems = validate_table(spec, table)
    checker = StructureChecker(spec, table, provenance.geometry_limits, workers=workers)
    part_sizes: list[int] = []

    def checked_parts() -> Iterator[Structures]:
        for part in structure_parts:
            checker.check(part)
            part_sizes.append(len(part))
            yield part

    staging = directory.with_name(f".{directory.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)  # left by an earlier run that died mid-write
    staging.mkdir(parents=True)
    try:
        structures_path = staging / STRUCTURES_FILENAME
        write_structures(structures_path, checked_parts())
        problems = table_problems + checker.problems()
        if problems:
            raise BundleValidationError(directory, problems)
        structure_file = StructureFile.open(structures_path)
        if len(structure_file) != len(table):
            raise BundleValidationError(
                directory,
                [
                    f"8: {STRUCTURES_FILENAME} holds {len(structure_file)} "
                    f"structures for {len(table)} table rows"
                ],
            )
        structure_hash = structures_sha256(
            structure_file,
            table["total_charge"].to_numpy(dtype="float64"),
            table["multiplicity"].to_numpy(dtype="float64"),
        )
        del structure_file
        written = write_table_files(
            staging,
            spec,
            table,
            provenance,
            structures_sha256=structure_hash,
            row_group_sizes=part_sizes,
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
