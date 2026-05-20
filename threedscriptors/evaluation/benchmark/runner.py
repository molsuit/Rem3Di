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


class BenchmarkResultRow(BaseModel):
    dataset_id: str
    source: Literal["moleculenet", "tdc"]
    descriptor_name: str
    learner_kind: str
    metric_name: str
    metric_value: float
    n_train: int
    n_val: int
    n_test: int


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
        if len(cols) != 1:
            raise ValueError(
                f"Multi-column regression benchmark not supported: "
                f"{[c.name for c in cols]}"
            )
        return "regression"
    if types == {TaskType.classification}:
        return "binary" if len(cols) == 1 else "multilabel"
    raise ValueError(f"Mixed task types in one benchmark: {types}")


def _dispatch_learner(
    learner: Learner,
    kind: Literal["regression", "binary", "multilabel"],
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Call the right ``fit_predict_*`` and return predictions on the test fold."""
    tr, va, te = splits
    Xtr, Xv, Xte = X[tr], X[va], X[te]
    Ytr, Yv = Y[tr], Y[va]
    if kind == "regression":
        return learner.fit_predict_regression(
            Xtr, Ytr[:, 0], Xv, Yv[:, 0], Xte, seed
        )
    if kind == "binary":
        return learner.fit_predict_binary(
            Xtr, Ytr[:, 0], Xv, Yv[:, 0], Xte, seed
        )
    return learner.fit_predict_multilabel(Xtr, Ytr, Xv, Yv, Xte, seed)


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


def evaluate_zarr(
    zarr_path: Path,
    manifest: BenchmarkManifest,
    cfg: EvalConfig,
    *,
    split_override: np.ndarray | None = None,
) -> list[BenchmarkResultRow]:
    dataset = MoleculeDataset.open_existing_dataset_from_dir(zarr_path)
    kind = _task_kind(dataset)
    Y = _build_targets_with_nan(dataset)
    tr_mask, va_mask, te_mask = _split_masks(dataset, split_override)
    Y_test_natural = Y[te_mask][:, 0] if kind != "multilabel" else Y[te_mask]
    cache_dir = cfg.descriptor_cache_dir or (cfg.output_dir / "descriptor_cache")

    rows: list[BenchmarkResultRow] = []
    for desc_cfg in cfg.descriptors:
        logger.info(
            "%s: computing descriptor %s", manifest.dataset_id, desc_cfg.name
        )
        X = compute_and_cache(desc_cfg, dataset, cache_dir, manifest.dataset_id)
        for learner_cfg in cfg.learners:
            learner = learner_cfg.build()
            preds = _dispatch_learner(
                learner, kind, (tr_mask, va_mask, te_mask), X, Y, cfg.seed
            )
            metric_value = _score(Y_test_natural, preds, manifest.metric)
            row = BenchmarkResultRow(
                dataset_id=manifest.dataset_id,
                source=manifest.source,
                descriptor_name=desc_cfg.name,
                learner_kind=_learner_kind(learner_cfg),
                metric_name=str(manifest.metric.value),
                metric_value=float(metric_value),
                n_train=int(tr_mask.sum()),
                n_val=int(va_mask.sum()),
                n_test=int(te_mask.sum()),
            )
            logger.info(
                "%s | %s | %s: %s=%.4f",
                manifest.dataset_id,
                desc_cfg.name,
                _learner_kind(learner_cfg),
                row.metric_name,
                row.metric_value,
            )
            rows.append(row)
    return rows


def run_eval(cfg: EvalConfig) -> list[BenchmarkResultRow]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[BenchmarkResultRow] = []
    for zarr_path, manifest in discover_benchmark_zarrs(cfg.eval_root):
        logger.info("--- %s @ %s ---", manifest.dataset_id, zarr_path)
        rows.extend(evaluate_zarr(zarr_path, manifest, cfg))
    write_results(rows, cfg.output_dir)
    return rows


def write_results(rows: list[BenchmarkResultRow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.model_dump() for r in rows])
    df.to_csv(output_dir / "results.csv", index=False)
    # Mirror the CSV as yaml for hand-inspection of small panels.
    pyd_yaml.to_yaml_file(
        output_dir / "results.yaml",
        _ResultBundle(rows=rows),
    )


class _ResultBundle(BaseModel):
    """Internal wrapper so pydantic_yaml can dump a list[BenchmarkResultRow]."""

    rows: list[BenchmarkResultRow]
