"""Benchmark eval runner: descriptor x learner cross-product over all zarrs.

Walks ``eval_config.eval_root`` via :func:`discover_benchmark_zarrs` — one
manifest per benchmark, no registry import. For each ``(dataset, descriptor,
learner)`` cell it:

1. Loads the stored ``split`` column (or applies the optional override) to
   partition rows into train / valid / test.
2. Computes descriptors over the full dataset (cached on disk by
   ``(dataset_id, descriptor.name)``) and slices them by split.
3. Dispatches to the matching ``Learner.fit_predict_*`` based on the
   benchmark's ``TaskSet`` shape — regression / binary / multilabel — passing
   train + valid + test (val drives early stopping where applicable).
4. Scores on test with ``manifest.metric``.

Results are pydantic ``BenchmarkResultRow`` objects; the runner writes them as
both a flat CSV and a yaml dump under ``output_dir``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pydantic_yaml as pyd_yaml
from pydantic import BaseModel, ConfigDict, Field

from threedscriptors.data_handling.benchmarks import (
    BenchmarkManifest,
    EvalMetric,
    discover_benchmark_zarrs,
)
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.tasks import Split, TaskType
from threedscriptors.evaluation.benchmark.descriptors import (
    DescriptorConfig,
    EcfpConfig,
    RemediConfig,
    compute_and_cache,
)
from threedscriptors.evaluation.benchmark.learners import (
    Learner,
    LearnerConfig,
)
from threedscriptors.evaluation.benchmark.metrics import metric_for

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
    # When True (default) a benchmark that raises is logged + recorded in
    # ``failures.yaml`` and the panel continues; ``results.csv`` is rewritten
    # after every dataset so a crash/timeout keeps everything completed so far.
    # Set False to fail fast on the first error.
    keep_going: bool = True


class BenchmarkResultRow(BaseModel):
    dataset_id: str
    source: Literal["moleculenet", "tdc", "polaris"]
    descriptor_name: str
    learner_kind: str
    # Which target column this row scores. Set per-column for regression
    # benchmarks (single- or multi-target); ``None`` for the macro-averaged
    # multilabel path.
    target_col: str | None
    metric_name: str
    metric_value: float
    n_train: int
    n_val: int
    n_test: int


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
) -> Literal["regression", "binary", "multilabel"]:
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


def _score(
    y_test: np.ndarray, y_pred: np.ndarray, metric: EvalMetric
) -> float:
    """Apply the manifest metric, masking single-target NaN rows for the
    1-D path. ``macro_auroc`` handles 2-D NaN columns on its own."""
    fn = metric_for(metric)
    if y_test.ndim == 1:
        keep = ~np.isnan(y_test)
        return fn(y_test[keep], y_pred[keep])
    return fn(y_test, y_pred)


def _split_masks(
    dataset: MoleculeDataset,
    override: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if override is not None:
        codes = np.asarray(override, dtype=np.uint8)
    else:
        if dataset.split is None:
            raise ValueError(
                "Dataset has no `split` column. Build it with the benchmark "
                "ingest runner so the literature split is materialized."
            )
        codes = np.asarray(dataset.split[:], dtype=np.uint8)
    return (
        codes == Split.train.value,
        codes == Split.valid.value,
        codes == Split.test.value,
    )


def _evaluate_cell(
    learner_cfg: LearnerConfig,
    kind: Literal["regression", "binary", "multilabel"],
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    col_names: list[str],
    manifest: BenchmarkManifest,
    descriptor_name: str,
    seed: int,
) -> list[BenchmarkResultRow]:
    """Evaluate one (descriptor x learner) cell, one row per scored target.

    Regression yields one row per target column (a fresh learner per column);
    binary and multilabel yield a single row.
    """
    tr, va, te = splits
    counts = dict(n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()))

    def make_row(metric_value: float, target_col: str | None) -> BenchmarkResultRow:
        return BenchmarkResultRow(
            dataset_id=manifest.dataset_id,
            source=manifest.source,
            descriptor_name=descriptor_name,
            learner_kind=_learner_kind(learner_cfg),
            target_col=target_col,
            metric_name=str(manifest.metric.value),
            metric_value=float(metric_value),
            **counts,
        )

    rows: list[BenchmarkResultRow] = []
    if kind == "multilabel":
        preds = learner_cfg.build().fit_predict_multilabel(
            X[tr], Y[tr], X[va], Y[va], X[te], seed
        )
        rows.append(make_row(_score(Y[te], preds, manifest.metric), None))
        return rows

    # regression / binary: score each target column independently.
    for col, name in enumerate(col_names):
        preds = _fit_predict_single_column(
            learner_cfg.build(), kind == "regression", splits, X, Y[:, col], seed
        )
        rows.append(make_row(_score(Y[te, col], preds, manifest.metric), name))
    return rows


def evaluate_zarr(
    zarr_path: Path,
    manifest: BenchmarkManifest,
    cfg: EvalConfig,
    *,
    split_override: np.ndarray | None = None,
) -> list[BenchmarkResultRow]:
    dataset = MoleculeDataset.open_existing_dataset_from_dir(zarr_path)
    kind = _task_kind(dataset)
    assert dataset.config.tasks is not None  # narrowed by _task_kind
    col_names = [c.name for c in dataset.config.tasks.system_cols]
    Y = _build_targets_with_nan(dataset)
    splits = _split_masks(dataset, split_override)
    cache_dir = cfg.descriptor_cache_dir or (cfg.output_dir / "descriptor_cache")

    rows: list[BenchmarkResultRow] = []
    for desc_cfg in cfg.descriptors:
        logger.info(
            "%s: computing descriptor %s", manifest.dataset_id, desc_cfg.name
        )
        X = compute_and_cache(desc_cfg, dataset, cache_dir, manifest.dataset_id)
        for learner_cfg in cfg.learners:
            cell_rows = _evaluate_cell(
                learner_cfg, kind, splits, X, Y, col_names, manifest,
                desc_cfg.name, cfg.seed,
            )
            for row in cell_rows:
                logger.info(
                    "%s | %s | %s | %s: %s=%.4f",
                    manifest.dataset_id,
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
    for zarr_path, manifest in discover_benchmark_zarrs(cfg.eval_root):
        logger.info("--- %s @ %s ---", manifest.dataset_id, zarr_path)
        try:
            rows.extend(evaluate_zarr(zarr_path, manifest, cfg))
        except Exception as exc:  # one bad benchmark must not sink the panel
            logger.exception(
                "benchmark %s failed; recording and continuing", manifest.dataset_id
            )
            failures.append(
                BenchmarkFailure(dataset_id=manifest.dataset_id, error=repr(exc))
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
