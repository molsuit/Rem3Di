"""Benchmark-panel task: descriptor x learner cross-product over benchmark zarrs.

The one panel loop. It opens each benchmark through
:mod:`remedi_eval.protocol` (partition, targets, task kind), scores it with the
per-cell helpers of :mod:`remedi_eval.benchmark.cells`, and sources descriptor
matrices from the shared :class:`ResourceCache` (so a baseline fingerprint / the
model embedding is computed once and reused).

Failures are isolated as narrowly as they happen: a benchmark that cannot be
opened, a descriptor matrix that cannot be loaded for one benchmark, and one
(descriptor x learner) cell that raises are each recorded in ``failures.yaml``
and the panel continues with everything else (with the manifest's
``keep_going: false`` the first failure is recorded and re-raised instead).
Both ``results.csv`` and ``failures.yaml`` are rewritten after every cell and
every failure, so a job killed by its wall-clock limit keeps what it did.

A multiclass cell also carries a :class:`MulticlassReport`; this task is what
serialises it, under ``benchmark/<dataset_id>/<descriptor>__<learner name>/`` as
``per_class.csv``, ``confusion_matrix.csv`` and ``confusion_matrix.npz``. The
npz is the plot input: :mod:`framework.builtin_plotters` registers a
row-normalised heatmap for the ``confusion_matrix`` artifact kind.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

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
from remedi_data.yaml_io import write_model
from remedi_eval.benchmark.cells import (
    BenchmarkCell,
    BenchmarkFailure,
    BenchmarkResultRow,
    discover_benchmarks,
    evaluate_cell,
    evaluate_pairwise_cell,
    multiclass_labelling,
)
from remedi_eval.benchmark.learners import LearnerConfig
from remedi_eval.benchmark.metrics import MulticlassReport
from remedi_eval.descriptors import DescriptorConfig
from remedi_eval.framework.context import EvalContext
from remedi_eval.framework.resources import EmbeddingSpec
from remedi_eval.protocol import Benchmark, Partition, TaskKind, load_benchmark

logger = logging.getLogger(__name__)

RESULTS_FILE_NAME = "results.csv"
FAILURES_FILE_NAME = "failures.yaml"


class _FailureBundle(BaseModel):
    failures: list[BenchmarkFailure]


class BenchmarkPanelConfig(BaseModel):
    """Property-prediction panel: every benchmark zarr x descriptor x learner."""

    kind: Literal["benchmark_panel"] = "benchmark_panel"
    eval_root: Path
    # Each learner's ``name`` (default: its ``learner_kind``) must be unique:
    # it is what tells two learners apart in results.csv and artifact paths.
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

    @field_validator("learners")
    @classmethod
    def _learner_names_are_unique(
        cls, learners: list[LearnerConfig]
    ) -> list[LearnerConfig]:
        names = [learner.name for learner in learners]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"learner names {duplicates} are used more than once; give each "
                "learner config a distinct `name`"
            )
        return learners

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
        record = _PanelRecord(ctx.task_dir("benchmark"), keep_going=ctx.keep_going)
        descriptors = [ctx.model, *self.baseline_descriptors]

        for zarr_path, spec in self.selected_benchmarks():
            logger.info("--- %s @ %s ---", spec.dataset_id, zarr_path)
            try:
                opened = self._open_benchmark(ctx, zarr_path, spec)
            except Exception as error:
                record.fail(
                    BenchmarkFailure(dataset_id=spec.dataset_id, error=repr(error)),
                    error,
                )
                continue
            for descriptor in descriptors:
                try:
                    features = ctx.resources.get(
                        EmbeddingSpec(
                            dataset_id=opened.benchmark.dataset_id,
                            descriptor=descriptor,
                            dataset=opened.benchmark.dataset,
                            cache_dir=ctx.resource_cache_dir,
                        )
                    )
                except Exception as error:
                    record.fail(
                        BenchmarkFailure(
                            dataset_id=spec.dataset_id,
                            descriptor_name=descriptor.name,
                            error=repr(error),
                        ),
                        error,
                    )
                    continue
                for learner_config in self.learners:
                    try:
                        rows, results = _evaluate_one_cell(
                            opened, features, descriptor.name, learner_config
                        )
                    except Exception as error:
                        record.fail(
                            BenchmarkFailure(
                                dataset_id=spec.dataset_id,
                                descriptor_name=descriptor.name,
                                learner_name=learner_config.name,
                                error=repr(error),
                            ),
                            error,
                        )
                        continue
                    record.add(rows)
                    yield from results

        yield TableResult(
            file_name=Path("benchmark") / RESULTS_FILE_NAME, frame=record.frame()
        )
        if record.failures:
            yield PydanticResult(
                file_name=Path("benchmark") / FAILURES_FILE_NAME,
                obj=_FailureBundle(failures=record.failures),
            )

    def _open_benchmark(
        self, ctx: EvalContext, zarr_path: Path, spec: DatasetSpec
    ) -> _OpenedBenchmark:
        """Everything one benchmark's cells share: targets, partition, cell identity.

        Raises whatever makes the whole benchmark unscorable (no table, an
        unknown split column, mixed task types) before any cell is fitted.
        """
        benchmark = load_benchmark(zarr_path)
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
        return _OpenedBenchmark(
            benchmark=benchmark,
            task_kind=benchmark.task_kind,
            targets=benchmark.targets(),
            partition=partition,
            cell=cell,
            stereoisomers=benchmark.stereoisomers() if is_pairwise else None,
        )


@dataclass(frozen=True)
class _OpenedBenchmark:
    """One opened benchmark and what every one of its cells is scored on.

    ``stereoisomers`` is set for the pairwise-ranking benchmark only.
    """

    benchmark: Benchmark
    task_kind: TaskKind
    targets: np.ndarray
    partition: Partition
    cell: BenchmarkCell
    stereoisomers: pd.DataFrame | None


@dataclass
class _PanelRecord:
    """The rows and failures so far, rewritten to disk on every change."""

    out_dir: Path
    keep_going: bool
    rows: list[BenchmarkResultRow] = field(default_factory=list)
    failures: list[BenchmarkFailure] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([row.model_dump() for row in self.rows])

    def add(self, rows: list[BenchmarkResultRow]) -> None:
        self.rows.extend(rows)
        self.frame().to_csv(self.out_dir / RESULTS_FILE_NAME, index=False)

    def fail(self, failure: BenchmarkFailure, error: Exception) -> None:
        """Record ``failure``; re-raise ``error`` unless the run keeps going."""
        logger.error(
            "benchmark %s (descriptor %s, learner %s) failed; recording",
            failure.dataset_id,
            failure.descriptor_name,
            failure.learner_name,
            exc_info=error,
        )
        self.failures.append(failure)
        write_model(
            self.out_dir / FAILURES_FILE_NAME, _FailureBundle(failures=self.failures)
        )
        if not self.keep_going:
            raise error


def _evaluate_one_cell(
    opened: _OpenedBenchmark,
    features: np.ndarray,
    descriptor_name: str,
    learner_config: LearnerConfig,
) -> tuple[list[BenchmarkResultRow], list[EvalResult]]:
    """One (descriptor x learner) cell: its result rows and its cell artifacts."""
    benchmark = opened.benchmark
    if opened.stereoisomers is not None:
        rows = evaluate_pairwise_cell(
            learner_config=learner_config,
            partition=opened.partition,
            X=features,
            Y=opened.targets,
            stereoisomers=opened.stereoisomers,
            col_name=benchmark.target_columns[0],
            cell=opened.cell,
            descriptor_name=descriptor_name,
        )
        return rows, []
    evaluation = evaluate_cell(
        learner_config=learner_config,
        kind=opened.task_kind,
        partition=opened.partition,
        X=features,
        Y=opened.targets,
        col_names=benchmark.target_columns,
        cell=opened.cell,
        descriptor_name=descriptor_name,
    )
    if evaluation.report is None:
        return evaluation.rows, []
    return evaluation.rows, _multiclass_artifacts(
        evaluation.report, benchmark.dataset_id, descriptor_name, learner_config.name
    )


def _multiclass_artifacts(
    report: MulticlassReport,
    dataset_id: str,
    descriptor_name: str,
    learner_name: str,
) -> list[EvalResult]:
    """The three per-cell artifacts of a multiclass cell's report.

    They land under ``benchmark/<dataset_id>/<descriptor>__<learner name>/`` so one
    panel run can hold a report per cell without any of them colliding. The
    confusion-matrix CSV is the human-readable form (a leading ``true_class``
    column, one column per predicted class); the npz is the machine-readable
    one the heatmap plotter re-renders from.
    """
    out_rel = Path("benchmark") / dataset_id / f"{descriptor_name}__{learner_name}"
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
