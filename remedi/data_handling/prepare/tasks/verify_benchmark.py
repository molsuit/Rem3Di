"""``verify_benchmark``: re-open a bundle and its zarr and re-assert the format.

Build-order step 4 of ``BENCHMARK_DATA_FORMAT.md``; §3 calls this "the Part-2
guard that actually asks whether the data is right". It is cheap and it runs
over data that is already on disk, so it is the one task worth running after
*every* prepare run and before every paper table.

Two halves:

* the **bundle** half is :func:`remedi.data_handling.bundle.read_bundle`
  itself, which re-validates invariants 1-10 and re-checks the hashes recorded
  in ``provenance.yaml`` on every open. ``conformer_timings.jsonl``, which
  ``generate_conformers`` writes into the bundle directory, is not part of the
  format and is ignored here exactly as ``read_bundle`` ignores it.
* the **zarr** half re-derives everything the ingest claimed: the four id
  arrays against the table's id columns, ``tasks/split`` against the codes of
  the default split column, and the copied ``table.parquet`` against the
  bundle's own content hash — a stale zarr left behind by an idempotent skip
  shows up as a content-hash mismatch and nothing else would catch it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from remedi.data_handling.bundle import (
    PROVENANCE_FILENAME,
    SPEC_FILENAME,
    TABLE_FILENAME,
    Bundle,
    content_hash_of_table,
    normalize_table,
    read_bundle,
    read_table,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset.tasks import split_codes_from_names
from remedi.data_handling.prepare.context import BundleRoots, PrepareContext
from remedi.data_handling.prepare.tasks.per_dataset import PerDatasetPrepareTask
from remedi.evaluation.results import EvalResult, PydanticResult

logger = logging.getLogger(__name__)


class VerifyBenchmarkReport(BaseModel):
    """What one dataset's verification found — the task's artifact."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    bundle_path: Path
    zarr_path: Path | None = None
    ok: bool = True
    rows: int = 0
    frames: int = 0
    structures_in_zarr: int = 0
    content_sha256: str | None = None
    problems: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


class VerifyBenchmarkConfig(PerDatasetPrepareTask):
    """Re-read every ``conformers``-stage bundle and its zarr, and re-assert both."""

    kind: Literal["verify_benchmark"] = "verify_benchmark"

    #: False verifies the bundle alone — useful before anything is ingested.
    check_zarr: bool = True

    # ----------------------------------------------------------------- running

    def discovery_root(self, roots: BundleRoots) -> Path:
        """This task verifies ``conformers``-stage bundles."""
        return roots.benchmark_root

    def run(self, ctx: PrepareContext) -> Iterator[EvalResult]:
        """Verify every resolved dataset id, reporting before failing.

        The report is yielded — and therefore written to disk — before the
        failure is raised, so ``<output_root>/verify_benchmark/<id>.yaml`` lists
        the problems even though ``status.yaml`` only records the traceback.

        Raises:
            FileNotFoundError: if a named bundle directory does not exist.
            BundleValidationError: if a bundle violates the format.
            ValueError: if the zarr disagrees with its bundle.
        """
        failed: list[str] = []
        for dataset_id in self.resolve_dataset_ids(ctx.roots()):
            report = self._verify_one_dataset(ctx, dataset_id)
            yield PydanticResult(
                file_name=Path("verify_benchmark") / f"{dataset_id}.yaml", obj=report
            )
            if not report.ok:
                failed.append(f"{dataset_id}: " + "; ".join(report.problems))
        if failed:
            raise ValueError(
                "verify_benchmark found problems:\n  " + "\n  ".join(failed)
            )

    def _verify_one_dataset(
        self, ctx: PrepareContext, dataset_id: str
    ) -> VerifyBenchmarkReport:
        started = time.monotonic()
        bundle_directory = Path(ctx.benchmark_root) / dataset_id
        # read_bundle re-validates invariants 1-10 and the recorded hashes; a
        # violation raises here and is recorded as a failed task.
        bundle = read_bundle(bundle_directory)

        zarr_path = Path(ctx.output_root) / dataset_id
        problems: list[str] = []
        structures_in_zarr = 0
        checked_zarr_path: Path | None = None
        if self.check_zarr and (zarr_path / "dataset_config.yaml").is_file():
            checked_zarr_path = zarr_path
            structures_in_zarr, problems = _verify_zarr(zarr_path, bundle)
        elif self.check_zarr:
            logger.info(
                "%s: no zarr at %s yet, verified the bundle only", dataset_id, zarr_path
            )

        report = VerifyBenchmarkReport(
            dataset_id=dataset_id,
            bundle_path=bundle_directory,
            zarr_path=checked_zarr_path,
            ok=not problems,
            rows=len(bundle.table),
            frames=len(bundle.structures or []),
            structures_in_zarr=structures_in_zarr,
            content_sha256=content_hash_of_table(bundle.table),
            problems=problems,
            elapsed_seconds=time.monotonic() - started,
        )
        logger.info(
            "%s: %s (%d rows, %d frames)",
            dataset_id,
            "ok" if report.ok else f"{len(problems)} problem(s)",
            report.rows,
            report.frames,
        )
        return report


def _verify_zarr(zarr_path: Path, bundle: Bundle) -> tuple[int, list[str]]:
    """Re-derive the ingest's claims about ``zarr_path``. Returns (rows, problems)."""
    dataset = MoleculeDataset.open_existing_dataset_from_dir(zarr_path)
    try:
        table = bundle.table
        n_rows = len(table)
        problems = _copied_bundle_problems(zarr_path, bundle)
        if dataset.N_structures != n_rows:
            problems.append(
                f"the zarr holds {dataset.N_structures} structures for "
                f"{n_rows} bundle rows"
            )
            return dataset.N_structures, problems
        problems += _id_array_problems(dataset, table, n_rows)
        problems += _split_problems(dataset, bundle, n_rows)
        return dataset.N_structures, problems
    finally:
        dataset.close()


def _copied_bundle_problems(zarr_path: Path, bundle: Bundle) -> list[str]:
    """The three bundle files must be beside the zarr and hold the same table."""
    problems: list[str] = []
    for filename in (SPEC_FILENAME, TABLE_FILENAME, PROVENANCE_FILENAME):
        if not (zarr_path / filename).is_file():
            problems.append(f"{filename} was not copied into the zarr directory")
    if (zarr_path / TABLE_FILENAME).is_file():
        copied = normalize_table(read_table(zarr_path), bundle.spec)
        copied_hash = content_hash_of_table(copied)
        bundle_hash = content_hash_of_table(bundle.table)
        if copied_hash != bundle_hash:
            problems.append(
                f"the copied {TABLE_FILENAME} has content_sha256 {copied_hash} but "
                f"the bundle's table has {bundle_hash} (a stale zarr?)"
            )
    return problems


def _id_array_problems(dataset: MoleculeDataset, table, n_rows: int) -> list[str]:
    """The four zarr id arrays against the bundle table's id columns (§2.3)."""
    expected_by_array = {
        "ids/structure_id": (dataset.structure_ids, np.arange(n_rows, dtype=np.int64)),
        "ids/bundle_row": (
            dataset.bundle_row,
            table["structure_id"].to_numpy(dtype=np.int64),
        ),
        "ids/molecule_id": (
            dataset.molecule_ids,
            table["molecule_id"].to_numpy(dtype=np.int64),
        ),
        "ids/stereoisomer_id": (
            dataset.isomer_ids,
            table["stereoisomer_id"].to_numpy(dtype=np.int64),
        ),
    }
    problems: list[str] = []
    for name, (array, expected) in expected_by_array.items():
        if array is None:
            problems.append(f"{name} is missing from the zarr")
        elif not np.array_equal(np.asarray(array[:n_rows], dtype=np.int64), expected):
            problems.append(f"{name} does not match the bundle table")
    return problems


def _split_problems(dataset: MoleculeDataset, bundle: Bundle, n_rows: int) -> list[str]:
    """``tasks/split`` must be the codes of the bundle's ``default_split`` column."""
    if dataset.split is None:
        return ["tasks/split is missing from the zarr"]
    expected = split_codes_from_names(
        bundle.table[bundle.spec.default_split].astype(str).tolist()
    )
    stored = np.asarray(dataset.split[:n_rows], dtype=np.uint8)
    if not np.array_equal(stored, expected):
        return [
            "tasks/split does not match the codes of the bundle's "
            f"{bundle.spec.default_split!r} column"
        ]
    return []
