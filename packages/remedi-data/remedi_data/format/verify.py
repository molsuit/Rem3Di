"""Re-check a dataset directory: table, structures, zarr and recorded hashes.

Also the two checks the writer shares with it: invariant 8 (the zarr against
the table, :func:`zarr_problems`) and the structures' hash
(:func:`structures_sha256`).

The guard that asks whether the data on disk is right. It runs over datasets
whichever path wrote them (``dataset_build`` or a preparer calling
``write_dataset`` directly) and is cheap enough to run before every paper table.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from remedi_data.dataset.molecule_dataset import MoleculeDataset
from remedi_data.format import BundleValidationError, read_bundle, structure_problems


def structures_sha256(dataset: MoleculeDataset) -> str:
    """sha256 over everything the model reads from the zarr.

    ``atomic_numbers``, ``positions`` and ``molecule_ptr`` (the structures) and
    the per-structure ``total_charge`` and ``multiplicity``. The data identity a
    descriptor cache and a selection key on: it moves whenever any structure,
    its charge or spin, or the set of structures changes.
    """
    digest = hashlib.sha256()
    for name, array in (
        ("atomic_numbers", dataset.atomic_numbers),
        ("positions", dataset.positions),
        ("molecule_ptr", dataset.ptr),
        ("total_charge", dataset.total_charge),
        ("multiplicity", dataset.multiplicity),
    ):
        values = np.ascontiguousarray(np.asarray(array[:]))
        digest.update(f"{name}:{values.dtype.str}:{values.shape}\n".encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def zarr_problems(path: Path, table: pd.DataFrame) -> list[str]:
    """Invariant 8: the zarr at ``path`` against the table, row for row.

    Structure count, the three id arrays, ``total_charge`` and ``multiplicity``
    must all equal what the table says. Labels and splits live only in the
    table, so there is nothing of theirs to compare.
    """
    dataset = MoleculeDataset.open_existing_dataset_from_dir(path)
    rows = len(table)
    if dataset.N_structures != rows:
        return [f"8: the zarr holds {dataset.N_structures} structures for {rows} rows"]
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
    for name, array in (
        ("total_charge", dataset.total_charge),
        ("multiplicity", dataset.multiplicity),
    ):
        if not np.array_equal(
            np.asarray(array[:rows], dtype=np.float64),
            table[name].to_numpy(dtype=np.float64),
        ):
            problems.append(f"8: {name} does not match the table")
    return problems


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
    try:
        dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_path)
    except (KeyError, OSError, ValueError) as error:
        # A missing array (KeyError), group or dataset_config.yaml (OSError) or
        # an unreadable config (ValueError): nothing below can run.
        report.problems.append(f"the zarr cannot be opened: {error!r}")
        report.elapsed_seconds = time.monotonic() - started
        return report
    report.problems += zarr_problems(dataset_path, bundle.table)
    report.structures_sha256 = structures_sha256(dataset)
    structures = dataset.read_structures()
    outputs = bundle.provenance.outputs
    recorded = None if outputs is None else outputs.structures_sha256
    if recorded != report.structures_sha256:
        report.problems.append(
            f"structures_sha256 of the zarr is {report.structures_sha256} but "
            f"provenance.yaml records {recorded}"
        )
    report.problems += structure_problems(
        bundle.spec,
        bundle.table,
        structures,
        bundle.provenance.geometry_limits,
        workers=workers,
    )
    report.elapsed_seconds = time.monotonic() - started
    return report
