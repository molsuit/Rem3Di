"""Re-check a dataset directory: table, structures, zarr and recorded hashes.

The guard that asks whether the data on disk is right. It runs over datasets
whichever path wrote them (``dataset_build`` or a preparer calling
``write_dataset`` directly) and is cheap enough to run before every paper table.
"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from remedi.data_handling.bundle import (
    BundleValidationError,
    read_bundle,
    structure_problems,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset_build.write import structures_sha256, zarr_problems


class VerifyReport(BaseModel):
    """What verifying one dataset found."""

    model_config = ConfigDict(extra="forbid")

    dataset_path: Path
    dataset_id: str | None = None
    rows: int = 0
    structures_sha256: str | None = None
    problems: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


def verify_dataset(dataset_path: Path) -> VerifyReport:
    """Every invariant on one dataset, collected rather than raised."""
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
    report.problems += zarr_problems(dataset_path, bundle.spec, bundle.table)
    dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_path)
    try:
        report.structures_sha256 = structures_sha256(dataset)
        structures = dataset.get_all_molecules()
    finally:
        dataset.close()
    outputs = bundle.provenance.outputs
    recorded = None if outputs is None else outputs.structures_sha256
    if recorded != report.structures_sha256:
        report.problems.append(
            f"structures_sha256 of the zarr is {report.structures_sha256} but "
            f"provenance.yaml records {recorded}"
        )
    report.problems += structure_problems(
        bundle.spec, bundle.table, structures, bundle.provenance.geometry_limits
    )
    report.elapsed_seconds = time.monotonic() - started
    return report
