"""Benchmark-panel task: descriptor x learner cross-product over benchmark datasets.

The one panel loop. It opens each benchmark through
:mod:`remedi_eval.protocol` (partition, targets, task kind), scores it with the
per-cell helpers of :mod:`remedi_eval.benchmark.cells`, and takes descriptor
matrices from :meth:`EvalContext.descriptor_matrix` (so a baseline fingerprint /
the model embedding is computed once and reused).

Failures are isolated as narrowly as they happen: a benchmark that cannot be
opened, a descriptor matrix that cannot be loaded for one benchmark, and one
(descriptor x learner) cell that raises are each recorded in ``failures.yaml``
and the panel continues with everything else (with the manifest's
``keep_going: false`` the first failure is recorded and re-raised instead).
``results.csv`` is yielded (and so rewritten once) after every cell and
``failures.yaml`` after every failure, so a job killed by its wall-clock limit
keeps what it did.

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

from remedi_data.format import DatasetSpec, EvalMetric
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
    multiclass_labelling,
)
from remedi_eval.benchmark.learners import Learner
from remedi_eval.benchmark.metrics import MulticlassReport
from remedi_eval.descriptors import DescriptorConfig
from remedi_eval.framework.context import EvalContext
from remedi_eval.protocol import Benchmark, Partition, TaskKind, load_benchmark

logger = logging.getLogger(__name__)

RESULTS_FILE_NAME = "results.csv"
FAILURES_FILE_NAME = "failures.yaml"


class _FailureBundle(BaseModel):
    failures: list[BenchmarkFailure]


class BenchmarkPanelConfig(BaseModel):
    """Property-prediction panel: every benchmark dataset x descriptor x learner."""

    kind: Literal["benchmark_panel"] = "benchmark_panel"
    eval_root: Path
    # Each learner's ``name`` (default: its ``kind``) must be unique:
    # it is what tells two learners apart in results.csv and artifact paths.
    learners: list[Learner] = Field(default_factory=list, min_length=1)
    # Descriptors evaluated in addition to the run's model (e.g. an ECFP
    # baseline). The run's model (``ctx.model``) is always evaluated first.
    baseline_descriptors: list[DescriptorConfig] = Field(default_factory=list)
    # The split columns of the bundle table to score on, each a separate cell
    # (e.g. ``split__seed1`` .. ``split__seed5``: a fixed test fold with five
    # train/valid partitions). The learner fitted on the i-th split runs with
    # seed ``manifest seed + i``, so the cells vary split and model seed
    # together. ``None`` scores each benchmark's own ``default_split`` once; a
    # name here must exist in every benchmark under ``eval_root``.
    split_columns: list[str] | None = Field(default=None, min_length=1)
    # Restrict the panel to these dataset ids; ``None`` scores every benchmark
    # under ``eval_root``. An id with no dataset under ``eval_root`` is an error.
    dataset_ids: list[str] | None = None

    @field_validator("split_columns")
    @classmethod
    def _split_columns_are_unique(cls, columns: list[str] | None) -> list[str] | None:
        if columns is not None and len(set(columns)) != len(columns):
            raise ValueError(f"split_columns {columns} repeat a column")
        return columns

    @field_validator("learners")
    @classmethod
    def _learner_names_are_unique(cls, learners: list[Learner]) -> list[Learner]:
        names = [learner.name for learner in learners]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"learner names {duplicates} are used more than once; give each "
                "learner config a distinct `name`"
            )
        return learners

    def selected_benchmarks(self) -> list[tuple[Path, DatasetSpec]]:
        """The ``(dataset_path, spec)`` pairs this panel scores, in discovery order.

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
        record = _PanelRecord(keep_going=ctx.keep_going)
        descriptors = [ctx.model, *self.baseline_descriptors]

        for dataset_path, spec in self.selected_benchmarks():
            logger.info("--- %s @ %s ---", spec.dataset_id, dataset_path)
            try:
                opened = self._open_benchmark(ctx, dataset_path, spec)
            except Exception as error:
                yield from record.fail(
                    BenchmarkFailure(dataset_id=spec.dataset_id, error=repr(error)),
                    error,
                )
                continue
            for descriptor in descriptors:
                try:
                    features = ctx.descriptor_matrix(
                        descriptor,
                        opened.benchmark.dataset,
                        opened.benchmark.dataset_id,
                    )
                except Exception as error:
                    yield from record.fail(
                        BenchmarkFailure(
                            dataset_id=spec.dataset_id,
                            descriptor_name=descriptor.name,
                            error=repr(error),
                        ),
                        error,
                    )
                    continue
                for learner in self.learners:
                    for split in opened.splits:
                        try:
                            rows, results = _evaluate_one_cell(
                                opened, split, features, descriptor.name, learner
                            )
                        except Exception as error:
                            yield from record.fail(
                                BenchmarkFailure(
                                    dataset_id=spec.dataset_id,
                                    descriptor_name=descriptor.name,
                                    learner_name=learner.name,
                                    error=repr(error),
                                ),
                                error,
                            )
                            continue
                        yield record.add(rows)
                        yield from results
        if not record.rows:
            yield record.table()

    def _open_benchmark(
        self, ctx: EvalContext, dataset_path: Path, spec: DatasetSpec
    ) -> _OpenedBenchmark:
        """Everything one benchmark's cells share: targets, partition, cell identity.

        Raises whatever makes the whole benchmark unscorable (no table, an
        unknown split column, mixed task types) before any cell is fitted.
        """
        benchmark = load_benchmark(dataset_path)
        n_classes, class_names = multiclass_labelling(spec)
        splits = []
        for index, split_column in enumerate(self.split_columns or [None]):
            partition = benchmark.partition(split_column)
            cell = BenchmarkCell(
                dataset_id=benchmark.dataset_id,
                metric=benchmark.headline_metric,
                split_column=partition.split_column,
                seed=ctx.seed + index,
                n_classes=n_classes,
                class_names=class_names,
            )
            splits.append(_ScoredSplit(partition=partition, cell=cell))
        is_pairwise = benchmark.headline_metric == EvalMetric.pair_ranking_accuracy
        return _OpenedBenchmark(
            benchmark=benchmark,
            task_kind=benchmark.task_kind,
            targets=benchmark.targets(),
            splits=splits,
            stereoisomers=benchmark.stereoisomers() if is_pairwise else None,
        )


@dataclass(frozen=True)
class _ScoredSplit:
    """One split column of a benchmark and the identity of its cells."""

    partition: Partition
    cell: BenchmarkCell


@dataclass(frozen=True)
class _OpenedBenchmark:
    """One opened benchmark and the splits its cells are scored on.

    ``stereoisomers`` is set for the pairwise-ranking benchmark only.
    """

    benchmark: Benchmark
    task_kind: TaskKind
    targets: np.ndarray
    splits: list[_ScoredSplit]
    stereoisomers: pd.DataFrame | None


@dataclass
class _PanelRecord:
    """The rows and failures so far.

    Every change comes back as the artifact to yield, so the runner rewrites
    ``results.csv`` or ``failures.yaml`` once per new cell or failure.
    """

    keep_going: bool
    rows: list[BenchmarkResultRow] = field(default_factory=list)
    failures: list[BenchmarkFailure] = field(default_factory=list)

    def table(self) -> TableResult:
        return TableResult(
            file_name=Path("benchmark") / RESULTS_FILE_NAME,
            frame=pd.DataFrame([row.model_dump() for row in self.rows]),
        )

    def add(self, rows: list[BenchmarkResultRow]) -> TableResult:
        self.rows.extend(rows)
        return self.table()

    def fail(self, failure: BenchmarkFailure, error: Exception) -> Iterator[EvalResult]:
        """Record ``failure`` and yield ``failures.yaml``; then re-raise
        ``error`` unless the run keeps going."""
        logger.error(
            "benchmark %s (descriptor %s, learner %s) failed; recording",
            failure.dataset_id,
            failure.descriptor_name,
            failure.learner_name,
            exc_info=error,
        )
        self.failures.append(failure)
        yield PydanticResult(
            file_name=Path("benchmark") / FAILURES_FILE_NAME,
            obj=_FailureBundle(failures=self.failures),
        )
        if not self.keep_going:
            raise error


def _evaluate_one_cell(
    opened: _OpenedBenchmark,
    split: _ScoredSplit,
    features: np.ndarray,
    descriptor_name: str,
    learner: Learner,
) -> tuple[list[BenchmarkResultRow], list[EvalResult]]:
    """One (descriptor x learner) cell: its result rows and its cell artifacts."""
    benchmark = opened.benchmark
    if opened.stereoisomers is not None:
        rows = evaluate_pairwise_cell(
            learner=learner,
            partition=split.partition,
            X=features,
            Y=opened.targets,
            stereoisomers=opened.stereoisomers,
            col_name=benchmark.target_columns[0],
            cell=split.cell,
            descriptor_name=descriptor_name,
        )
        return rows, []
    evaluation = evaluate_cell(
        learner=learner,
        kind=opened.task_kind,
        partition=split.partition,
        X=features,
        Y=opened.targets,
        col_names=benchmark.target_columns,
        cell=split.cell,
        descriptor_name=descriptor_name,
    )
    if evaluation.report is None:
        return evaluation.rows, []
    return evaluation.rows, _multiclass_artifacts(
        evaluation.report,
        benchmark.dataset_id,
        descriptor_name,
        learner.name,
        split.cell.split_column,
    )


def _multiclass_artifacts(
    report: MulticlassReport,
    dataset_id: str,
    descriptor_name: str,
    learner_name: str,
    split_column: str,
) -> list[EvalResult]:
    """The three per-cell artifacts of a multiclass cell's report.

    They land under
    ``benchmark/<dataset_id>/<descriptor>__<learner name>/<split column>/`` so
    one panel run can hold a report per cell without any of them colliding. The
    confusion-matrix CSV is the human-readable form (a leading ``true_class``
    column, one column per predicted class); the npz is the machine-readable
    one the heatmap plotter re-renders from.
    """
    out_rel = (
        Path("benchmark")
        / dataset_id
        / f"{descriptor_name}__{learner_name}"
        / split_column
    )
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
