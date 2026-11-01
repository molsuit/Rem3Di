"""Scoring one benchmark cell: (dataset, descriptor, learner) on one partition.

The framework's ``benchmark_panel`` task
(:mod:`remedi_eval.framework.tasks.benchmark`) composes these into a panel. The
partition, targets and task kind come from :mod:`remedi_eval.protocol`;
:func:`evaluate_cell` dispatches to the matching ``Learner.fit_predict_*``
from the task kind, passing train + valid + test (valid drives early stopping
where applicable), and scores the test fold with the spec's headline metric.

Results are pydantic :class:`BenchmarkResultRow` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from remedi_data.bundle import DatasetSpec, EvalMetric, discover_datasets
from remedi_data.dataset.tasks import TaskType
from remedi_eval.benchmark.learners import (
    Learner,
    LearnerConfig,
)
from remedi_eval.benchmark.metrics import (
    MulticlassReport,
    metric_for,
    multiclass_report,
)
from remedi_eval.benchmark.pairwise import pair_ranking_accuracy
from remedi_eval.protocol import Partition, TaskKind


class BenchmarkResultRow(BaseModel):
    """One scored cell. Provenance beyond these fields lives in the bundle."""

    dataset_id: str
    descriptor_name: str
    learner_kind: str
    # Which target column this row scores. Set per-column for regression
    # benchmarks (single- or multi-target); ``None`` for the macro-averaged
    # multilabel path.
    target_col: str | None
    metric_name: str
    metric_value: float
    # Which partition produced these folds and which seed the learner ran
    # with: without both, five seed runs of one dataset are indistinguishable
    # rows (§2.4).
    split_column: str
    seed: int
    n_train: int
    n_val: int
    n_test: int


@dataclass(frozen=True)
class BenchmarkCell:
    """What identifies one scored cell, beyond the descriptor and the learner.

    ``n_classes`` / ``class_names`` are the multiclass labelling the benchmark
    spec declares (:class:`LabelColumn`). They carry no identity — they only
    say how the cell's per-class report is shaped and headed — and both are
    ``None`` for every other task kind. With ``n_classes`` unset the multiclass
    path falls back to the class count observed in the labels.
    """

    dataset_id: str
    metric: EvalMetric
    split_column: str
    seed: int
    n_classes: int | None = None
    class_names: tuple[str, ...] | None = None


@dataclass
class CellEvaluation:
    """What one (descriptor x learner) cell produced.

    ``rows`` is what lands in ``results.csv`` — one per scored target column.
    ``report`` is set on a multiclass cell only: the per-class table and
    confusion matrix over the test fold, which the panel task writes out as its
    own artifacts (the flat result row cannot carry them).
    """

    rows: list[BenchmarkResultRow]
    report: MulticlassReport | None = None


class BenchmarkFailure(BaseModel):
    """One benchmark that raised during the panel (recorded, not fatal)."""

    dataset_id: str
    error: str


def learner_kind(cfg: LearnerConfig) -> str:
    return cfg.learner_kind  # type: ignore[union-attr]


def _fit_predict_single_column(
    learner: Learner,
    is_regression: bool,
    partition: Partition,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fit one single-target column, dropping NaN-label rows from train/val.

    Sparse multi-target regression (e.g. polaris_adme_fang) leaves NaN where a
    molecule has no measurement for that target; Ridge / the MLP can't fit on
    NaN labels, so the missing rows are filtered out of the fit folds. Test
    predictions are still produced over the full test fold — ``_score`` masks
    the NaN test rows when scoring.
    """
    tr, va, te = partition.train, partition.valid, partition.test
    ytr, yva = y[tr], y[va]
    tr_keep, va_keep = ~np.isnan(ytr), ~np.isnan(yva)
    Xtr, Xva, Xte = X[tr][tr_keep], X[va][va_keep], X[te]
    ytr, yva = ytr[tr_keep], yva[va_keep]
    if is_regression:
        return learner.fit_predict_regression(Xtr, ytr, Xva, yva, Xte, seed)
    return learner.fit_predict_binary(Xtr, ytr, Xva, yva, Xte, seed)


def _score(y_test: np.ndarray, y_pred: np.ndarray, metric: EvalMetric) -> float:
    """Apply the manifest metric, masking single-target NaN rows for the
    1-D path. ``macro_auroc`` handles 2-D NaN columns on its own."""
    fn = metric_for(metric)
    if y_test.ndim == 1:
        keep = ~np.isnan(y_test)
        return fn(y_test[keep], y_pred[keep])
    return fn(y_test, y_pred)


def evaluate_cell(
    learner_cfg: LearnerConfig,
    kind: TaskKind,
    partition: Partition,
    X: np.ndarray,
    Y: np.ndarray,
    col_names: list[str],
    cell: BenchmarkCell,
    descriptor_name: str,
) -> CellEvaluation:
    """Evaluate one (descriptor x learner) cell, one row per scored target.

    Regression yields one row per target column (a fresh learner per column);
    binary, multilabel and multiclass each yield a single row. A multiclass cell
    additionally carries its :class:`MulticlassReport` over the test fold.
    """
    tr, va, te = partition.train, partition.valid, partition.test
    seed = cell.seed
    counts = dict(n_train=len(tr), n_val=len(va), n_test=len(te))

    def make_row(metric_value: float, target_col: str | None) -> BenchmarkResultRow:
        return BenchmarkResultRow(
            dataset_id=cell.dataset_id,
            descriptor_name=descriptor_name,
            learner_kind=learner_kind(learner_cfg),
            target_col=target_col,
            metric_name=str(cell.metric.value),
            metric_value=float(metric_value),
            split_column=cell.split_column,
            seed=seed,
            **counts,
        )

    rows: list[BenchmarkResultRow] = []
    if kind == "multilabel":
        preds = learner_cfg.build().fit_predict_multilabel(
            X[tr], Y[tr], X[va], Y[va], X[te], seed
        )
        rows.append(make_row(_score(Y[te], preds, cell.metric), None))
        return CellEvaluation(rows=rows)

    if kind == "multiclass":
        # One column of integer class indices. The spec's declared class count
        # wins; without it the count spans the whole dataset (the stratified
        # split guarantees every class is present).
        y = Y[:, 0]
        n_classes = cell.n_classes or int(np.nanmax(y)) + 1
        preds = learner_cfg.build().fit_predict_multiclass(
            X[tr], y[tr], X[va], y[va], X[te], n_classes, seed
        )
        rows.append(make_row(_score(y[te], preds, cell.metric), None))
        # Every multiclass cell gets the per-class picture, not just the
        # headline number the flat leaderboard can carry (§6).
        report = multiclass_report(
            y[te].astype(int), preds, n_classes, cell.class_names
        )
        return CellEvaluation(rows=rows, report=report)

    # regression / binary: score each target column independently.
    for col, name in enumerate(col_names):
        preds = _fit_predict_single_column(
            learner_cfg.build(), kind == "regression", partition, X, Y[:, col], seed
        )
        rows.append(make_row(_score(Y[te, col], preds, cell.metric), name))
    return CellEvaluation(rows=rows)


def evaluate_pairwise_cell(
    learner_cfg: LearnerConfig,
    partition: Partition,
    X: np.ndarray,
    Y: np.ndarray,
    molecule_ids: np.ndarray,
    isomer_ids: np.ndarray,
    col_name: str,
    cell: BenchmarkCell,
    descriptor_name: str,
) -> list[BenchmarkResultRow]:
    """Evaluate one (descriptor x learner) cell for the pairwise ranking task.

    The single regression column (the docking ``top_score``) is fit as an
    ordinary regression; the test-fold predictions are then handed to
    :func:`pair_ranking_accuracy`, which pools conformers per stereoisomer and
    ranks each enantiomer pair. One result row, with ``n_test`` carrying the
    number of enantiomer pairs scored (not conformers).
    """
    te = partition.test
    y = Y[:, 0]
    preds = _fit_predict_single_column(
        learner_cfg.build(), True, partition, X, y, cell.seed
    )
    acc, n_pairs = pair_ranking_accuracy(y[te], preds, molecule_ids[te], isomer_ids[te])
    return [
        BenchmarkResultRow(
            dataset_id=cell.dataset_id,
            descriptor_name=descriptor_name,
            learner_kind=learner_kind(learner_cfg),
            target_col=col_name,
            metric_name=str(cell.metric.value),
            metric_value=float(acc),
            split_column=cell.split_column,
            seed=cell.seed,
            n_train=len(partition.train),
            n_val=len(partition.valid),
            n_test=int(n_pairs),
        )
    ]


def discover_benchmarks(root: Path) -> list[tuple[Path, DatasetSpec]]:
    """Every dataset directly under ``root`` that declares an evaluation block."""
    return [(path, spec) for path, spec in discover_datasets(root) if spec.evaluation]


def multiclass_labelling(
    spec: DatasetSpec,
) -> tuple[int | None, tuple[str, ...] | None]:
    """The declared class count and display names of a multiclass benchmark.

    ``(None, None)`` for every benchmark that is not single-column multiclass —
    the multiclass path then falls back to the class count it observes in the
    labels and to ``class_0 … class_{n-1}`` names.
    """
    multiclass_tasks = [
        task for task in spec.labels if task.task_type is TaskType.multiclass
    ]
    if len(multiclass_tasks) != 1:
        return None, None
    task = multiclass_tasks[0]
    names = tuple(task.class_names) if task.class_names is not None else None
    return task.n_classes, names
