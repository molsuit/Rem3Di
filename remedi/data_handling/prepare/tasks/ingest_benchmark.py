"""``ingest_benchmark``: bundle(``conformers``) -> zarr.

Build-order step 4 of ``BENCHMARK_DATA_FORMAT.md``. This is what is left of
``build_benchmark_dataset.py`` once filtering has moved into the preparer
(§2.2) and conformer generation into ``generate_conformers`` (§1.2): read one
prepared bundle, stream it through ``[CopyDataStage]`` alone, and write the
zarr the eval side opens.

Three things make this more than a rename of the old script:

* **The pipeline is chosen by nothing.** A ``conformers``-stage bundle always
  ingests through ``[CopyDataStage]``; the chain of ``isinstance`` checks that
  picked a generator and a filter per source is gone with the five generators.
* **The ids are carried, not recomputed** (§2.3). The dataset is created with
  ``contains_smiles=False``, so ``get_mol_ids_for_batch`` copies the bundle's
  ``molecule_id`` / ``stereoisomer_id`` / ``structure_id`` verbatim and the
  writer stores the bundle row in ``ids/bundle_row``.
* **The bundle travels with the zarr.** ``benchmark.yaml``, ``table.parquet``
  and ``provenance.yaml`` are copied into the zarr directory, which makes the
  zarr self-describing: the eval side reads the spec (metrics, tasks, split
  columns) and every split column and SMILES from the table, joined through
  ``ids/bundle_row`` (§1a of the step-4 plan).
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from remedi.configuration.dataset_config import DatasetConfig, DatasetCreationConfig
from remedi.data_handling.bundle import (
    PROVENANCE_FILENAME,
    SPEC_FILENAME,
    TABLE_FILENAME,
    Bundle,
    content_hash_of_table,
    read_bundle,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset_creation.generators import (
    PreparedBenchmarkGenerator,
)
from remedi.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from remedi.data_handling.dataset_creation.pipeline_stages import CopyDataStage
from remedi.data_handling.prepare.context import BundleRoots, PrepareContext
from remedi.data_handling.prepare.tasks.per_dataset import PerDatasetPrepareTask
from remedi.evaluation.results import EvalResult, PydanticResult

logger = logging.getLogger(__name__)

#: The bundle files copied into the zarr directory so it is self-describing.
COPIED_BUNDLE_FILES: tuple[str, ...] = (
    SPEC_FILENAME,
    TABLE_FILENAME,
    PROVENANCE_FILENAME,
)


class IngestBenchmarkSummary(BaseModel):
    """What one dataset's ingest did — the task's artifact."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    bundle_path: Path
    zarr_path: Path
    skipped: bool = False
    rows: int = 0
    n_atoms: int = 0
    task_names: list[str] = Field(default_factory=list)
    default_split: str = ""
    rows_per_split: dict[str, int] = Field(default_factory=dict)
    #: The bundle table's logical content hash (§1.4) — the comparability key,
    #: and the only way to notice that an idempotent skip kept a stale zarr.
    content_sha256: str | None = None
    elapsed_seconds: float = 0.0


class IngestBenchmarkConfig(PerDatasetPrepareTask):
    """Write one zarr per ``conformers``-stage bundle under ``benchmark_root``.

    The zarr lands at ``PrepareContext.output_root / <dataset_id>``, beside the
    run's ``manifest.yaml`` / ``status.yaml``; that directory is what an eval
    manifest points ``eval_root`` at.
    """

    kind: Literal["ingest_benchmark"] = "ingest_benchmark"

    #: Bundle rows per :class:`InputBatch` handed to the pipeline.
    batch_size: int = Field(default=256, ge=1)
    #: False skips a dataset whose zarr directory already exists.
    overwrite: bool = False

    # zarr chunk / shard geometry, mirroring ``DatasetConfig``'s defaults.
    atom_chunk: int = Field(default=8192, ge=1)
    molecule_chunk: int = Field(default=4096, ge=1)
    atom_chunks_per_shard: int = Field(default=64, ge=1)
    molecule_chunks_per_shard: int = Field(default=256, ge=1)

    # ----------------------------------------------------------------- running

    def discovery_root(self, roots: BundleRoots) -> Path:
        """This task consumes ``conformers``-stage bundles."""
        return roots.benchmark_root

    def dataset_config(self, bundle: Bundle) -> DatasetConfig:
        """The zarr schema for ``bundle``: the bundle's tasks, no SMILES store.

        ``contains_smiles=False`` is what makes the orchestrator copy the
        bundle's ids instead of re-deriving them from a SMILES store (§2.3);
        the SMILES themselves stay readable through the copied
        ``table.parquet``.
        """
        return DatasetConfig(
            atom_chunk=self.atom_chunk,
            molecule_chunk=self.molecule_chunk,
            atom_chunks_per_shard=self.atom_chunks_per_shard,
            molecule_chunks_per_shard=self.molecule_chunks_per_shard,
            contains_smiles=False,
            tasks=bundle.spec.task_set(),
        )

    def run(self, ctx: PrepareContext) -> Iterator[EvalResult]:
        """Ingest every resolved dataset id into ``ctx.output_root``.

        Raises:
            FileNotFoundError: if a named bundle directory does not exist.
            BundleValidationError: if a bundle violates the format.
            ValueError: if the bundle is not at the ``conformers`` stage, or if
                the written zarr fails the post-build id gate.
        """
        for dataset_id in self.resolve_dataset_ids(ctx.roots()):
            yield self._run_one_dataset(ctx, dataset_id)

    def _run_one_dataset(self, ctx: PrepareContext, dataset_id: str) -> EvalResult:
        started = time.monotonic()
        bundle_directory = Path(ctx.benchmark_root) / dataset_id
        zarr_path = Path(ctx.output_root) / dataset_id
        summary_path = Path("ingest_benchmark") / f"{dataset_id}.yaml"

        if zarr_path.exists() and not self.overwrite:
            logger.info(
                "%s: %s already exists, skipping (overwrite is False)",
                dataset_id,
                zarr_path,
            )
            return PydanticResult(
                file_name=summary_path,
                obj=IngestBenchmarkSummary(
                    dataset_id=dataset_id,
                    bundle_path=bundle_directory,
                    zarr_path=zarr_path,
                    skipped=True,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )

        bundle = read_bundle(bundle_directory)
        generator = PreparedBenchmarkGenerator(bundle, batch_size=self.batch_size)
        if zarr_path.exists():
            # Rebuilding in place would leave the previous run's copied bundle
            # files behind if the new bundle dropped a column.
            shutil.rmtree(zarr_path)

        logger.info(
            "%s: ingesting %d rows -> %s", dataset_id, generator.n_rows, zarr_path
        )
        DatasetConstructionOrchestrator(
            pipeline=[CopyDataStage(dtype=torch.float64)],
            batch_generator=generator,
            construction_config=DatasetCreationConfig(path=zarr_path),
            dataset_config=self.dataset_config(bundle),
        ).build_dataset()

        n_atoms = _check_written_ids(zarr_path, bundle)
        for filename in COPIED_BUNDLE_FILES:
            shutil.copyfile(bundle_directory / filename, zarr_path / filename)

        spec = bundle.spec
        return PydanticResult(
            file_name=summary_path,
            obj=IngestBenchmarkSummary(
                dataset_id=dataset_id,
                bundle_path=bundle_directory,
                zarr_path=zarr_path,
                rows=len(bundle.table),
                n_atoms=n_atoms,
                task_names=spec.task_names(),
                default_split=spec.default_split,
                rows_per_split={
                    str(value): int(count)
                    for value, count in bundle.table[spec.default_split]
                    .value_counts()
                    .items()
                },
                content_sha256=content_hash_of_table(bundle.table),
                elapsed_seconds=time.monotonic() - started,
            ),
        )


def _check_written_ids(zarr_path: Path, bundle: Bundle) -> int:
    """Assert the zarr's id arrays agree with the bundle, and return its atom count.

    ``ids/structure_id`` must stay ``arange(N)`` for a ``[CopyDataStage]``-only
    ingest, because ``MoleculeDataset.get_structure_ids_from_molecule_ids`` is
    consumed positionally by the training-side splitter. A bundle whose
    ``structure_id`` column is not the row index would break that silently, so
    it is checked here rather than assumed.

    Raises:
        ValueError: on any disagreement, naming every one it found.
    """
    dataset = MoleculeDataset.open_existing_dataset_from_dir(zarr_path)
    try:
        table = bundle.table
        n_rows = len(table)
        problems: list[str] = []
        if dataset.N_structures != n_rows:
            problems.append(
                f"the zarr holds {dataset.N_structures} structures for "
                f"{n_rows} bundle rows"
            )
        else:
            expected_by_array = {
                "ids/structure_id": (
                    dataset.structure_ids,
                    np.arange(n_rows, dtype=np.int64),
                ),
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
            for name, (array, expected) in expected_by_array.items():
                if array is None:
                    problems.append(f"{name} is missing from the written zarr")
                elif not np.array_equal(
                    np.asarray(array[:n_rows], dtype=np.int64), expected
                ):
                    problems.append(f"{name} does not match the bundle table")
        if problems:
            raise ValueError(
                f"ingest_benchmark wrote an inconsistent zarr at {zarr_path}:\n  "
                + "\n  ".join(problems)
            )
        return dataset.N_atoms
    finally:
        dataset.close()
