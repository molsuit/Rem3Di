"""Scoring one benchmark cell: (dataset, descriptor, learner) on one split.

The building blocks the framework's ``benchmark_panel`` task
(:mod:`remedi.evaluation.framework.tasks.benchmark`) composes into a panel:

1. :func:`split_masks` reads the train / valid / test partition **by name from
   the bundle's ``table.parquet``**, which travels with the zarr, joined through
   ``ids/bundle_row``. That is what lets a run pick a non-default split (a seed
   variant) without re-ingesting anything; the zarr's own ``tasks/split`` array
   only ever holds the bundle's ``default_split``.
2. :func:`build_targets_with_nan` gives the label matrix with NaN for missing
   labels and each spec label's transform applied.
3. :func:`evaluate_cell` dispatches to the matching ``Learner.fit_predict_*``
   from the benchmark's task kind (:func:`task_kind`), passing train + valid +
   test (valid drives early stopping where applicable), and scores the test
   fold with the spec's headline metric (``spec.evaluation.metrics[0]``).

Results are pydantic :class:`BenchmarkResultRow` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from remedi.data_handling.bundle import (
    TABLE_FILENAME,
    DatasetSpec,
    EvalMetric,
    EvaluationSpec,
    discover_datasets,
    read_table,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset.tasks import Split, TaskType, split_codes_from_names
from remedi.evaluation.benchmark.learners import (
    Learner,
    LearnerConfig,
)
from remedi.evaluation.benchmark.metrics import (
    MulticlassReport,
    metric_for,
    multiclass_report,
)
from remedi.evaluation.benchmark.pairwise import pair_ranking_accuracy


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


def apply_label_transforms(
    targets: np.ndarray, column_names: list[str], spec: DatasetSpec
) -> np.ndarray:
    """Each target column mapped through its spec label's ``transform``.

    ``targets`` is ``(n_structures, n_columns)`` with columns named by
    ``column_names``; NaN (missing) stays NaN. Every column must be a label the
    spec declares, so a transform can never be silently skipped.
    """
    if targets.ndim != 2 or targets.shape[1] != len(column_names):
        raise ValueError(
            f"targets of shape {targets.shape} do not match the "
            f"{len(column_names)} target columns {column_names}"
        )
    labels_by_name = {label.name: label for label in spec.labels}
    transformed = np.array(targets, dtype=float, copy=True)
    for column_index, column_name in enumerate(column_names):
        label = labels_by_name.get(column_name)
        if label is None:
            raise ValueError(
                f"target column {column_name!r} is not a label of "
                f"{spec.dataset_id} (labels: {spec.label_names()})"
            )
        transformed[:, column_index] = label.transform.apply(
            transformed[:, column_index]
        )
    return transformed


def build_targets_with_nan(dataset: MoleculeDataset, spec: DatasetSpec) -> np.ndarray:
    """``targets_system`` with NaN where ``mask_system == 0``, label transforms applied.

    Downstream learner code uses NaN to skip missing labels in sparse multi-
    label benchmarks (SIDER / ClinTox / Tox21). The zarr holds raw labels; each
    column goes through its spec label's ``transform`` here, the single place
    targets enter fitting and scoring, so metrics are on the transformed scale.
    """
    if dataset.targets_system is None or dataset.mask_system is None:
        raise ValueError(
            "Dataset has no system task arrays; cannot evaluate. Build with "
            "the benchmark ingest runner so targets are materialized."
        )
    if dataset.config.tasks is None:
        raise ValueError("Dataset has no TaskSet; cannot name its target columns.")
    targets = np.asarray(dataset.targets_system[:], dtype=float)
    mask = np.asarray(dataset.mask_system[:], dtype=bool)
    column_names = [column.name for column in dataset.config.tasks.system_cols]
    return apply_label_transforms(np.where(mask, targets, np.nan), column_names, spec)


def task_kind(
    dataset: MoleculeDataset,
) -> Literal["regression", "binary", "multilabel", "multiclass"]:
    if dataset.config.tasks is None:
        raise ValueError("Dataset has no TaskSet; cannot infer task kind.")
    cols = dataset.config.tasks.system_cols
    types = {c.task_type for c in cols}
    if types == {TaskType.regression}:
        # Multi-target regression (e.g. polaris_adme_fang) is scored per column
        # downstream — one independent single-target fit per target.
        return "regression"
    if types == {TaskType.classification}:
        return "binary" if len(cols) == 1 else "multilabel"
    if types == {TaskType.multiclass}:
        # Single-label multi-class: exactly one column carrying integer class
        # indices (e.g. chiral_cat's 5 chirality types).
        if len(cols) != 1:
            raise ValueError(
                f"multiclass benchmark must have one column, got {len(cols)}"
            )
        return "multiclass"
    raise ValueError(f"Mixed task types in one benchmark: {types}")


def _fit_predict_single_column(
    learner: Learner,
    is_regression: bool,
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
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
    tr, va, te = splits
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


def split_masks(
    zarr_path: Path,
    dataset: MoleculeDataset,
    split_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train / valid / test boolean masks over the zarr's structures.

    The split is read by name from the ``table.parquet`` that ``ingest_benchmark``
    copied beside the zarr and gathered onto the zarr's rows through
    ``ids/bundle_row``, so any of the bundle's split columns can be scored, not
    just the one materialised into ``tasks/split``.

    Raises:
        ValueError: if the zarr carries no table, if ``split_column`` is not one
            of its columns, if its ``structure_id`` column is not the row index,
            or if a split value is not a known split name.
    """
    n_structures = dataset.N_structures
    try:
        table = read_table(zarr_path, columns=["structure_id", split_column])
    except FileNotFoundError as error:
        raise ValueError(
            f"{zarr_path} has no {TABLE_FILENAME}; re-ingest it with the "
            "ingest_benchmark prepare task, which copies the bundle beside the zarr"
        ) from error
    except Exception as error:  # pyarrow raises its own type for a missing column
        available = [
            name for name in read_table(zarr_path).columns if name.startswith("split")
        ]
        raise ValueError(
            f"split column {split_column!r} is not in {zarr_path / TABLE_FILENAME}; "
            f"its split columns are {available}"
        ) from error

    structure_ids = table["structure_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(structure_ids, np.arange(len(table), dtype=np.int64)):
        raise ValueError(
            f"{zarr_path / TABLE_FILENAME} violates invariant 1: structure_id is not "
            "the row index, so predictions cannot be joined back to it"
        )
    rows = np.asarray(
        dataset.bundle_rows_or_structure_ids[:n_structures], dtype=np.int64
    )
    codes = split_codes_from_names(table[split_column].to_numpy(dtype=object)[rows])
    return (
        codes == Split.train.value,
        codes == Split.valid.value,
        codes == Split.test.value,
    )


def evaluate_cell(
    learner_cfg: LearnerConfig,
    kind: Literal["regression", "binary", "multilabel", "multiclass"],
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
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
    tr, va, te = splits
    seed = cell.seed
    counts = dict(n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()))

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
            learner_cfg.build(), kind == "regression", splits, X, Y[:, col], seed
        )
        rows.append(make_row(_score(Y[te, col], preds, cell.metric), name))
    return CellEvaluation(rows=rows)


def evaluate_pairwise_cell(
    learner_cfg: LearnerConfig,
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
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
    tr, va, te = splits
    y = Y[:, 0]
    preds = _fit_predict_single_column(
        learner_cfg.build(), True, splits, X, y, cell.seed
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
            n_train=int(tr.sum()),
            n_val=int(va.sum()),
            n_test=int(n_pairs),
        )
    ]


def structure_group_ids(
    dataset: MoleculeDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-structure (constitution id, stereoisomer id) aligned to descriptor rows."""
    return (
        np.asarray(dataset.molecule_ids[:]),
        np.asarray(dataset.isomer_ids[:]),
    )


def evaluation_of(spec: DatasetSpec) -> EvaluationSpec:
    """The spec's evaluation block.

    Raises:
        ValueError: if the dataset declares none (a corpus, not a benchmark).
    """
    if spec.evaluation is None:
        raise ValueError(f"{spec.dataset_id} declares no evaluation block")
    return spec.evaluation


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
