"""Benchmark eval runner: descriptor x learner cross-product over all zarrs.

Walks ``eval_config.eval_root`` via :func:`discover_benchmark_zarrs` — one
:class:`BenchmarkSpec` per benchmark, copied into the zarr at ingest, no
registry import. For each ``(dataset, descriptor, learner)`` cell it:

1. Reads the split column **by name from the bundle's ``table.parquet``**,
   which travels with the zarr, joined through ``ids/bundle_row``. That is
   what lets a run pick a non-default split (a seed variant, a scaffold
   variant) without re-ingesting anything; the zarr's own ``tasks/split``
   array only ever holds the bundle's ``default_split``.
2. Computes descriptors over the full dataset (cached on disk by
   ``(dataset_id, descriptor.name)``) and slices them by split.
3. Dispatches to the matching ``Learner.fit_predict_*`` based on the
   benchmark's ``TaskSet`` shape — regression / binary / multilabel — passing
   train + valid + test (val drives early stopping where applicable).
4. Scores on test with the spec's headline metric (``spec.metrics[0]``).

Results are pydantic ``BenchmarkResultRow`` objects; the runner writes them as
both a flat CSV and a yaml dump under ``output_dir``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pydantic_yaml as pyd_yaml
from pydantic import BaseModel, ConfigDict, Field

from remedi.data_handling.bundle import (
    TABLE_FILENAME,
    BenchmarkSpec,
    EvalMetric,
    discover_benchmark_zarrs,
    read_table,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset.tasks import Split, TaskType, split_codes_from_names
from remedi.evaluation.benchmark.descriptors import (
    DescriptorConfig,
    EcfpConfig,
    RemediConfig,
    compute_and_cache,
)
from remedi.evaluation.benchmark.learners import (
    Learner,
    LearnerConfig,
)
from remedi.evaluation.benchmark.metrics import metric_for
from remedi.evaluation.benchmark.pairwise import pair_ranking_accuracy

logger = logging.getLogger(__name__)


class EvalConfig(BaseModel):
    """One yaml = the full eval panel (every prepared zarr xevery learner)."""

    model_config = ConfigDict(extra="forbid")

    eval_root: Path
    output_dir: Path
    descriptors: list[DescriptorConfig] = Field(default_factory=list, min_length=1)
    learners: list[LearnerConfig] = Field(default_factory=list, min_length=1)
    # Defaults to ``output_dir / "descriptor_cache"`` when omitted.
    descriptor_cache_dir: Path | None = None
    # Forwarded to learner.fit_predict_* methods that accept a seed.
    seed: int = 0
    # Which split column of the bundle table to score on. ``None`` uses each
    # benchmark's own ``default_split``; a name here must exist in every
    # benchmark under ``eval_root``.
    split_column: str | None = None
    # When True (default) a benchmark that raises is logged + recorded in
    # ``failures.yaml`` and the panel continues; ``results.csv`` is rewritten
    # after every dataset so a crash/timeout keeps everything completed so far.
    # Set False to fail fast on the first error.
    keep_going: bool = True


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
    """What identifies one scored cell, beyond the descriptor and the learner."""

    dataset_id: str
    metric: EvalMetric
    split_column: str
    seed: int


class BenchmarkFailure(BaseModel):
    """One benchmark that raised during the panel (recorded, not fatal)."""

    dataset_id: str
    error: str


def _descriptor_name(cfg: EcfpConfig | RemediConfig) -> str:
    return cfg.name


def _learner_kind(cfg: LearnerConfig) -> str:
    return cfg.learner_kind  # type: ignore[union-attr]


def _build_targets_with_nan(dataset: MoleculeDataset) -> np.ndarray:
    """``targets_system`` with NaN where ``mask_system == 0``.

    Downstream learner code uses NaN to skip missing labels in sparse multi-
    label benchmarks (SIDER / ClinTox / Tox21).
    """
    if dataset.targets_system is None or dataset.mask_system is None:
        raise ValueError(
            "Dataset has no system task arrays; cannot evaluate. Build with "
            "the benchmark ingest runner so targets are materialized."
        )
    targets = np.asarray(dataset.targets_system[:], dtype=float)
    mask = np.asarray(dataset.mask_system[:], dtype=bool)
    return np.where(mask, targets, np.nan)


def _task_kind(
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


def _split_masks(
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


def _evaluate_cell(
    learner_cfg: LearnerConfig,
    kind: Literal["regression", "binary", "multilabel", "multiclass"],
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    col_names: list[str],
    cell: BenchmarkCell,
    descriptor_name: str,
) -> list[BenchmarkResultRow]:
    """Evaluate one (descriptor x learner) cell, one row per scored target.

    Regression yields one row per target column (a fresh learner per column);
    binary, multilabel and multiclass each yield a single row.
    """
    tr, va, te = splits
    seed = cell.seed
    counts = dict(n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()))

    def make_row(metric_value: float, target_col: str | None) -> BenchmarkResultRow:
        return BenchmarkResultRow(
            dataset_id=cell.dataset_id,
            descriptor_name=descriptor_name,
            learner_kind=_learner_kind(learner_cfg),
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
        return rows

    if kind == "multiclass":
        # One column of integer class indices; the class count spans the whole
        # dataset (the stratified split guarantees every class is present).
        y = Y[:, 0]
        n_classes = int(np.nanmax(y)) + 1
        preds = learner_cfg.build().fit_predict_multiclass(
            X[tr], y[tr], X[va], y[va], X[te], n_classes, seed
        )
        rows.append(make_row(_score(y[te], preds, cell.metric), None))
        return rows

    # regression / binary: score each target column independently.
    for col, name in enumerate(col_names):
        preds = _fit_predict_single_column(
            learner_cfg.build(), kind == "regression", splits, X, Y[:, col], seed
        )
        rows.append(make_row(_score(Y[te, col], preds, cell.metric), name))
    return rows


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
            learner_kind=_learner_kind(learner_cfg),
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


def _structure_group_ids(
    dataset: MoleculeDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-structure (constitution id, stereoisomer id) aligned to descriptor rows."""
    return (
        np.asarray(dataset.molecule_ids[:]),
        np.asarray(dataset.isomer_ids[:]),
    )


def evaluate_zarr(
    zarr_path: Path,
    spec: BenchmarkSpec,
    cfg: EvalConfig,
    *,
    split_column: str | None = None,
) -> list[BenchmarkResultRow]:
    """Every (descriptor x learner) cell of one prepared benchmark zarr."""
    dataset = MoleculeDataset.open_existing_dataset_from_dir(zarr_path)
    kind = _task_kind(dataset)
    assert dataset.config.tasks is not None  # narrowed by _task_kind
    col_names = [c.name for c in dataset.config.tasks.system_cols]
    Y = _build_targets_with_nan(dataset)
    cell = BenchmarkCell(
        dataset_id=spec.dataset_id,
        # The first metric is the reported cell; the rest are computed at table
        # time from the cached predictions (§1.3, §4.1).
        metric=spec.metrics[0],
        split_column=split_column or cfg.split_column or spec.default_split,
        seed=cfg.seed,
    )
    splits = _split_masks(zarr_path, dataset, cell.split_column)
    cache_dir = cfg.descriptor_cache_dir or (cfg.output_dir / "descriptor_cache")

    is_pairwise = cell.metric == EvalMetric.pair_ranking_accuracy
    mol_ids, iso_ids = _structure_group_ids(dataset) if is_pairwise else (None, None)

    rows: list[BenchmarkResultRow] = []
    for desc_cfg in cfg.descriptors:
        logger.info("%s: computing descriptor %s", spec.dataset_id, desc_cfg.name)
        X = compute_and_cache(desc_cfg, dataset, cache_dir, spec.dataset_id)
        for learner_cfg in cfg.learners:
            if is_pairwise:
                assert mol_ids is not None and iso_ids is not None
                cell_rows = evaluate_pairwise_cell(
                    learner_cfg,
                    splits,
                    X,
                    Y,
                    mol_ids,
                    iso_ids,
                    col_names[0],
                    cell,
                    desc_cfg.name,
                )
            else:
                cell_rows = _evaluate_cell(
                    learner_cfg,
                    kind,
                    splits,
                    X,
                    Y,
                    col_names,
                    cell,
                    desc_cfg.name,
                )
            for row in cell_rows:
                logger.info(
                    "%s | %s | %s | %s: %s=%.4f",
                    spec.dataset_id,
                    desc_cfg.name,
                    _learner_kind(learner_cfg),
                    row.target_col,
                    row.metric_name,
                    row.metric_value,
                )
            rows.extend(cell_rows)
    return rows


def run_eval(cfg: EvalConfig) -> list[BenchmarkResultRow]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[BenchmarkResultRow] = []
    failures: list[BenchmarkFailure] = []
    for zarr_path, spec in discover_benchmark_zarrs(cfg.eval_root):
        logger.info("--- %s @ %s ---", spec.dataset_id, zarr_path)
        try:
            rows.extend(evaluate_zarr(zarr_path, spec, cfg))
        except Exception as exc:  # one bad benchmark must not sink the panel
            logger.exception(
                "benchmark %s failed; recording and continuing", spec.dataset_id
            )
            failures.append(
                BenchmarkFailure(dataset_id=spec.dataset_id, error=repr(exc))
            )
            if not cfg.keep_going:
                write_results(rows, cfg.output_dir, failures)
                raise
        # Persist after every dataset so a later crash / wall-clock timeout
        # keeps everything completed so far (descriptors stay cached, so a
        # resubmit only redoes the cheap probe fits for what's missing).
        write_results(rows, cfg.output_dir, failures)
    if failures:
        logger.warning(
            "benchmark panel finished with %d failed dataset(s): %s",
            len(failures),
            ", ".join(f.dataset_id for f in failures),
        )
    return rows


def write_results(
    rows: list[BenchmarkResultRow],
    output_dir: Path,
    failures: list[BenchmarkFailure] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.model_dump() for r in rows])
    df.to_csv(output_dir / "results.csv", index=False)
    # Mirror the CSV as yaml for hand-inspection of small panels.
    pyd_yaml.to_yaml_file(
        output_dir / "results.yaml",
        _ResultBundle(rows=rows),
    )
    if failures:
        pyd_yaml.to_yaml_file(
            output_dir / "failures.yaml", _FailureBundle(failures=failures)
        )


class _ResultBundle(BaseModel):
    """Internal wrapper so pydantic_yaml can dump a list[BenchmarkResultRow]."""

    rows: list[BenchmarkResultRow]


class _FailureBundle(BaseModel):
    """Internal wrapper so pydantic_yaml can dump a list[BenchmarkFailure]."""

    failures: list[BenchmarkFailure]
