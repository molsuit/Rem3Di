"""Benchmark-panel task: descriptor x learner cross-product over benchmark zarrs.

The one panel loop. It opens each benchmark through
:mod:`remedi_eval.protocol` (partition, targets, task kind), scores it with the
per-cell helpers of :mod:`remedi_eval.benchmark.cells`, and sources descriptor matrices
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

from remedi_data.bundle import (
    DatasetSpec,
    EvalMetric,
)
from remedi_data.runs.results import (
    ArrayResult,
    EvalResult,
    PydanticResult,
    TableResult,
)
from remedi_eval.benchmark.cells import (
    BenchmarkCell,
    BenchmarkFailure,
    BenchmarkResultRow,
    discover_benchmarks,
    evaluate_cell,
    evaluate_pairwise_cell,
    learner_kind,
    multiclass_labelling,
)
from remedi_eval.benchmark.learners import LearnerConfig
from remedi_eval.benchmark.metrics import MulticlassReport
from remedi_eval.descriptors import DescriptorConfig
from remedi_eval.framework.context import EvalContext
from remedi_eval.framework.resources import EmbeddingSpec
from remedi_eval.protocol import load_benchmark

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
    # Restrict the panel to these dataset ids; ``None`` scores every benchmark
    # under ``eval_root``. An id with no zarr under ``eval_root`` is an error.
    dataset_ids: list[str] | None = None

    def selected_benchmarks(self) -> list[tuple[Path, DatasetSpec]]:
        """The ``(zarr_path, spec)`` pairs this panel scores, in discovery order.

        Raises:
            ValueError: if a requested dataset id is not under ``eval_root``.
        """
        discovered = list(discover_benchmarks(self.eval_root))
        if self.dataset_ids is None:
            return discovered
        available = {spec.dataset_id for _, spec in discovered}
        missing = sorted(set(self.dataset_ids) - available)
        if missing:
            raise ValueError(
                f"dataset ids {missing} have no benchmark under {self.eval_root}"
            )
        wanted = set(self.dataset_ids)
        return [(path, spec) for path, spec in discovered if spec.dataset_id in wanted]

    def run(self, ctx: EvalContext) -> Iterator[EvalResult]:
        out = ctx.task_dir("benchmark")
        descriptors = [ctx.model, *self.baseline_descriptors]
        rows: list[BenchmarkResultRow] = []
        failures: list[BenchmarkFailure] = []

        for zarr_path, spec in self.selected_benchmarks():
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
        spec: DatasetSpec,
        descriptors: list[DescriptorConfig],
    ) -> tuple[list[BenchmarkResultRow], list[EvalResult]]:
        """Every cell of one benchmark: its result rows and its cell artifacts."""
        benchmark = load_benchmark(zarr_path)
        Y = benchmark.targets()
        n_classes, class_names = multiclass_labelling(spec)
        partition = benchmark.partition(self.split_column)
        cell = BenchmarkCell(
            dataset_id=benchmark.dataset_id,
            metric=benchmark.headline_metric,
            split_column=partition.split_column,
            seed=ctx.seed,
            n_classes=n_classes,
            class_names=class_names,
        )
        is_pairwise = cell.metric == EvalMetric.pair_ranking_accuracy
        mol_ids, iso_ids = (
            benchmark.structure_group_ids() if is_pairwise else (None, None)
        )
        rows: list[BenchmarkResultRow] = []
        results: list[EvalResult] = []
        for desc in descriptors:
            X = ctx.resources.get(
                EmbeddingSpec(
                    dataset_id=benchmark.dataset_id,
                    descriptor=desc,
                    dataset=benchmark.dataset,
                    cache_dir=ctx.resource_cache_dir,
                )
            )
            for learner_cfg in self.learners:
                if is_pairwise:
                    assert mol_ids is not None and iso_ids is not None
                    rows.extend(
                        evaluate_pairwise_cell(
                            learner_cfg,
                            partition,
                            X,
                            Y,
                            mol_ids,
                            iso_ids,
                            benchmark.target_columns[0],
                            cell,
                            desc.name,
                        )
                    )
                    continue
                evaluation = evaluate_cell(
                    learner_cfg,
                    benchmark.task_kind,
                    partition,
                    X,
                    Y,
                    benchmark.target_columns,
                    cell,
                    desc.name,
                )
                rows.extend(evaluation.rows)
                if evaluation.report is not None:
                    results.extend(
                        _multiclass_artifacts(
                            evaluation.report,
                            benchmark.dataset_id,
                            desc.name,
                            learner_kind(learner_cfg),
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
