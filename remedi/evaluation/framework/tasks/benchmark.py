"""Benchmark-panel task: descriptor x learner cross-product over benchmark zarrs.

The framework port of ``benchmark.runner.run_eval``. It reuses that module's
per-cell evaluation helpers verbatim (``_evaluate_cell``, ``_task_kind``,
``_build_targets_with_nan``, ``_split_masks``) but sources descriptor matrices
from the shared :class:`ResourceCache` (so a baseline fingerprint / the model
embedding is computed once and reused) and keeps the fault-tolerant, incremental
write behaviour: one failing benchmark is recorded in ``failures.yaml`` and the
panel continues, with ``results.csv`` rewritten after every dataset.

A multiclass cell also carries a :class:`MulticlassReport`; this task is what
serialises it, under ``benchmark/<dataset_id>/<descriptor>__<learner>/`` as
``per_class.csv``, ``confusion_matrix.csv`` and ``confusion_matrix.npz``. The
npz is the plot input: :mod:`framework.builtin_plotters` registers a
row-normalised heatmap for the ``confusion_matrix`` artifact kind.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from remedi.data_handling.bundle import (
    BenchmarkSpec,
    EvalMetric,
    discover_benchmark_zarrs,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.evaluation.benchmark.descriptors import DescriptorConfig
from remedi.evaluation.benchmark.learners import LearnerConfig
from remedi.evaluation.benchmark.metrics import MulticlassReport
from remedi.evaluation.benchmark.runner import (
    BenchmarkCell,
    BenchmarkFailure,
    BenchmarkResultRow,
    _build_targets_with_nan,
    _evaluate_cell,
    _learner_kind,
    _split_masks,
    _structure_group_ids,
    _task_kind,
    evaluate_pairwise_cell,
    multiclass_labelling,
)
from remedi.evaluation.framework.context import EvalContext
from remedi.evaluation.framework.resources import EmbeddingSpec
from remedi.evaluation.results import (
    ArrayResult,
    EvalResult,
    PydanticResult,
    TableResult,
)

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
    # Which split column of the bundle table to score on. ``None`` uses each
    # benchmark's own ``default_split``; a name here must exist in every
    # benchmark under ``eval_root``.
    split_column: str | None = None

    def run(self, ctx: EvalContext) -> Iterator[EvalResult]:
        out = ctx.task_dir("benchmark")
        descriptors = [ctx.model, *self.baseline_descriptors]
        rows: list[BenchmarkResultRow] = []
        failures: list[BenchmarkFailure] = []

        for zarr_path, spec in discover_benchmark_zarrs(self.eval_root):
            logger.info("--- %s @ %s ---", spec.dataset_id, zarr_path)
            dataset_rows: list[BenchmarkResultRow] = []
            dataset_results: list[EvalResult] = []
            try:
                dataset_rows, dataset_results = self._evaluate_dataset(
                    ctx, zarr_path, spec, descriptors
                )
            except Exception as exc:  # one bad benchmark must not sink the panel
                logger.exception("benchmark %s failed; recording", spec.dataset_id)
                failures.append(
                    BenchmarkFailure(dataset_id=spec.dataset_id, error=repr(exc))
                )
            rows.extend(dataset_rows)
            # Crash-safe incremental write after each dataset.
            self._write_csv(rows, out)
            yield from dataset_results

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
        self,
        ctx: EvalContext,
        zarr_path: Path,
        spec: BenchmarkSpec,
        descriptors: list[DescriptorConfig],
    ) -> tuple[list[BenchmarkResultRow], list[EvalResult]]:
        """Every cell of one benchmark: its result rows and its cell artifacts."""
        dataset = MoleculeDataset.open_existing_dataset_from_dir(zarr_path)
        kind = _task_kind(dataset)
        assert dataset.config.tasks is not None  # narrowed by _task_kind
        col_names = [c.name for c in dataset.config.tasks.system_cols]
        Y = _build_targets_with_nan(dataset)
        n_classes, class_names = multiclass_labelling(spec)
        cell = BenchmarkCell(
            dataset_id=spec.dataset_id,
            metric=spec.metrics[0],
            split_column=self.split_column or spec.default_split,
            seed=ctx.seed,
            n_classes=n_classes,
            class_names=class_names,
        )
        splits = _split_masks(zarr_path, dataset, cell.split_column)
        is_pairwise = cell.metric == EvalMetric.pair_ranking_accuracy
        mol_ids, iso_ids = (
            _structure_group_ids(dataset) if is_pairwise else (None, None)
        )
        rows: list[BenchmarkResultRow] = []
        results: list[EvalResult] = []
        for desc in descriptors:
            X = ctx.resources.get(
                EmbeddingSpec(
                    dataset_id=spec.dataset_id,
                    descriptor=desc,
                    dataset=dataset,
                    cache_dir=ctx.resource_cache_dir,
                )
            )
            for learner_cfg in self.learners:
                if is_pairwise:
                    assert mol_ids is not None and iso_ids is not None
                    rows.extend(
                        evaluate_pairwise_cell(
                            learner_cfg,
                            splits,
                            X,
                            Y,
                            mol_ids,
                            iso_ids,
                            col_names[0],
                            cell,
                            desc.name,
                        )
                    )
                    continue
                evaluation = _evaluate_cell(
                    learner_cfg,
                    kind,
                    splits,
                    X,
                    Y,
                    col_names,
                    cell,
                    desc.name,
                )
                rows.extend(evaluation.rows)
                if evaluation.report is not None:
                    results.extend(
                        _multiclass_artifacts(
                            evaluation.report,
                            spec.dataset_id,
                            desc.name,
                            _learner_kind(learner_cfg),
                        )
                    )
        return rows, results

    @staticmethod
    def _frame(rows: list[BenchmarkResultRow]) -> pd.DataFrame:
        return pd.DataFrame([r.model_dump() for r in rows])

    def _write_csv(self, rows: list[BenchmarkResultRow], out_dir: Path) -> None:
        self._frame(rows).to_csv(out_dir / "results.csv", index=False)


def _multiclass_artifacts(
    report: MulticlassReport,
    dataset_id: str,
    descriptor_name: str,
    learner_kind: str,
) -> list[EvalResult]:
    """The three per-cell artifacts of a multiclass cell's report.

    They land under ``benchmark/<dataset_id>/<descriptor>__<learner>/`` so one
    panel run can hold a report per cell without any of them colliding. The
    confusion-matrix CSV is the human-readable form (a leading ``true_class``
    column, one column per predicted class); the npz is the machine-readable
    one the heatmap plotter re-renders from.
    """
    out_rel = Path("benchmark") / dataset_id / f"{descriptor_name}__{learner_kind}"
    names = report.class_names
    confusion_frame = pd.DataFrame(report.confusion_matrix, columns=names)
    confusion_frame.insert(0, "true_class", names)
    return [
        TableResult(file_name=out_rel / "per_class.csv", frame=report.per_class),
        TableResult(file_name=out_rel / "confusion_matrix.csv", frame=confusion_frame),
        ArrayResult(
            file_name=out_rel / "confusion_matrix.npz",
            arrays={
                "confusion_matrix": report.confusion_matrix,
                # The display names, not the integer indices: they are what the
                # registered heatmap plotter labels its axes with.
                "labels": np.asarray(names, dtype=np.str_),
            },
        ),
    ]
