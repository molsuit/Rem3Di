"""Re-check a dataset directory: table, structures and recorded hashes.

The guard that asks whether the data on disk is right. It runs over datasets
whichever path wrote them (``dataset_build`` or a preparer calling
``write_dataset`` directly) and is cheap enough to run before every paper table.
It reads the structures through :mod:`remedi_data.dataset`, which itself reads
the format, so this module is imported by its path and not exported by
:mod:`remedi_data.format`.
"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from remedi_data.dataset.dataset import Dataset
from remedi_data.dataset.structure_file import READ_CHUNK_STRUCTURES
from remedi_data.format.invariants import structure_problems
from remedi_data.format.table import (
    PROVENANCE_FILENAME,
    STRUCTURES_FILENAME,
    BundleValidationError,
    read_bundle,
)


class VerifyReport(BaseModel):
    """What verifying one dataset found."""

    model_config = ConfigDict(extra="forbid")

    dataset_path: Path
    dataset_id: str | None = None
    rows: int = 0
    structures_sha256: str | None = None
    problems: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


def verify_dataset(dataset_path: Path, *, workers: int = 1) -> VerifyReport:
    """Every invariant on one dataset, collected rather than raised.

    The table invariants and its recorded hash (:func:`read_bundle`), one
    record in ``structures.atp`` per table row, the recorded
    ``structures_sha256``, and invariants 9 and 10 on every frame (read in
    pieces, so memory stays bounded; ``workers`` processes check them).
    """
    started = time.monotonic()
    dataset_path = Path(dataset_path)
    report = VerifyReport(dataset_path=dataset_path)
    try:
        bundle = read_bundle(dataset_path)
    except (BundleValidationError, FileNotFoundError) as error:
        report.problems.append(str(error))
        report.elapsed_seconds = time.monotonic() - started
        return report
    report.dataset_id = bundle.spec.dataset_id
    report.rows = len(bundle.table)
    if not bundle.spec.has_structures:
        report.problems.append("dataset.yaml declares no geometry_origin")
    try:
        dataset = Dataset.open(dataset_path)
    except (BundleValidationError, OSError, ValueError) as error:
        # A missing or unreadable structures.atp, or one whose record count is
        # not the table's (BundleValidationError): nothing below can run.
        report.problems.append(f"{STRUCTURES_FILENAME} cannot be used: {error!r}")
        report.elapsed_seconds = time.monotonic() - started
        return report
    report.structures_sha256 = dataset.structures_sha256()
    outputs = bundle.provenance.outputs
    recorded = None if outputs is None else outputs.structures_sha256
    if recorded != report.structures_sha256:
        report.problems.append(
            f"structures_sha256 of {STRUCTURES_FILENAME} and the table is "
            f"{report.structures_sha256} but {PROVENANCE_FILENAME} records {recorded}"
        )
    report.problems += structure_problems(
        bundle.spec,
        bundle.table,
        (
            dataset.structures.slice(start, start + READ_CHUNK_STRUCTURES)
            for start in range(0, len(dataset), READ_CHUNK_STRUCTURES)
        ),
        bundle.provenance.geometry_limits,
        workers=workers,
    )
    report.elapsed_seconds = time.monotonic() - started
    return report
