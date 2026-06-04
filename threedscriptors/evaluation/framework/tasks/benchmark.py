"""Benchmark-panel task: descriptor x learner cross-product over benchmark zarrs.

The framework port of ``benchmark.runner.run_eval``. It reuses that module's
per-cell evaluation helpers verbatim (``_evaluate_cell``, ``_task_kind``,
``_build_targets_with_nan``, ``_split_masks``) but sources descriptor matrices
from the shared :class:`ResourceCache` (so a baseline fingerprint / the model
embedding is computed once and reused) and keeps the fault-tolerant, incremental
write behaviour: one failing benchmark is recorded in ``failures.yaml`` and the
panel continues, with ``results.csv`` rewritten after every dataset.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from threedscriptors.data_handling.benchmarks import discover_benchmark_zarrs
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.evaluation.benchmark.descriptors import DescriptorConfig
from threedscriptors.evaluation.benchmark.learners import LearnerConfig
from threedscriptors.evaluation.benchmark.runner import (
    BenchmarkFailure,
    BenchmarkResultRow,
    _build_targets_with_nan,
    _evaluate_cell,
    _split_masks,
    _task_kind,
)
from threedscriptors.evaluation.framework.context import EvalContext
from threedscriptors.evaluation.framework.resources import EmbeddingSpec
from threedscriptors.evaluation.results import EvalResult, PydanticResult, TableResult

logger = logging.getLogger(__name__)


class _FailureBundle(BaseModel):
    failures: list[BenchmarkFailure]


class BenchmarkPanelConfig(BaseModel):
    """Property-prediction panel: every benchmark zarr x descriptor x learner."""

    kind: Literal["benchmark_panel"] = "benchmark_panel"
    eval_root: Path
    learners: list[LearnerConfig] = Field(default_factory=list, min_length=1)
    # Descriptors evaluated in addition to the run's model (e.g. an ECFP
    # baseline). The run's model (``ctx.model``) is always evaluated first.
    baseline_descriptors: list[DescriptorConfig] = Field(default_factory=list)

    def run(self, ctx: EvalContext) -> Iterator[EvalResult]:
        out = ctx.task_dir("benchmark")
        descriptors = [ctx.model, *self.baseline_descriptors]
        rows: list[BenchmarkResultRow] = []
        failures: list[BenchmarkFailure] = []

        for zarr_path, manifest in discover_benchmark_zarrs(self.eval_root):
            logger.info("--- %s @ %s ---", manifest.dataset_id, zarr_path)
            try:
                rows.extend(
                    self._evaluate_dataset(ctx, zarr_path, manifest, descriptors)
                )
            except Exception as exc:  # one bad benchmark must not sink the panel
                logger.exception("benchmark %s failed; recording", manifest.dataset_id)
                failures.append(
                    BenchmarkFailure(dataset_id=manifest.dataset_id, error=repr(exc))
                )
            # Crash-safe incremental write after each dataset.
            self._write_csv(rows, out)

        yield TableResult(
            file_name=Path("benchmark/results.csv"),
            frame=self._frame(rows),
        )
        if failures:
            yield PydanticResult(
                file_name=Path("benchmark/failures.yaml"),
                obj=_FailureBundle(failures=failures),
            )

    def _evaluate_dataset(
        self, ctx: EvalContext, zarr_path: Path, manifest, descriptors
    ) -> list[BenchmarkResultRow]:
        dataset = MoleculeDataset.open_existing_dataset_from_dir(zarr_path)
        kind = _task_kind(dataset)
        assert dataset.config.tasks is not None  # narrowed by _task_kind
        col_names = [c.name for c in dataset.config.tasks.system_cols]
        Y = _build_targets_with_nan(dataset)
        splits = _split_masks(dataset, None)
        rows: list[BenchmarkResultRow] = []
        for desc in descriptors:
            X = ctx.resources.get(
                EmbeddingSpec(
                    dataset_id=manifest.dataset_id,
                    descriptor=desc,
                    dataset=dataset,
                    cache_dir=ctx.resource_cache_dir,
                )
            )
            for learner_cfg in self.learners:
                rows.extend(
                    _evaluate_cell(
                        learner_cfg, kind, splits, X, Y, col_names,
                        manifest, desc.name, ctx.seed,
                    )
                )
        return rows

    @staticmethod
    def _frame(rows: list[BenchmarkResultRow]) -> pd.DataFrame:
        return pd.DataFrame([r.model_dump() for r in rows])

    def _write_csv(self, rows: list[BenchmarkResultRow], out_dir: Path) -> None:
        self._frame(rows).to_csv(out_dir / "results.csv", index=False)
