"""Metric functions for the descriptor-probe benchmark."""

from __future__ import annotations

import numpy as np

from threedscriptors.data_handling.benchmarks import EvalMetric
from threedscriptors.evaluation.benchmark.metrics import (
    auprc,
    auroc,
    macro_auroc,
    mae,
    metric_for,
    r2,
    rmse,
    spearman,
)


def test_regression_metrics_on_known_inputs() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y, yhat) == 0.0
    assert mae(y, yhat) == 0.0
    assert r2(y, yhat) == 1.0
    # Spearman is rank-based; a monotone transform should still give rho=1.
    assert abs(spearman(y, np.array([10.0, 20.0, 30.0, 40.0])) - 1.0) < 1e-9


def test_perfectly_separated_binary_auroc_one() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc(y, p) == 1.0
    assert auprc(y, p) == 1.0


def test_macro_auroc_skips_single_class_columns() -> None:
    # Column 0 perfectly separable (AUROC=1), column 1 is all zeros (skip).
    y = np.array([[0, 0], [0, 0], [1, 0], [1, 0]], dtype=float)
    p = np.array([[0.1, 0.5], [0.2, 0.5], [0.8, 0.5], [0.9, 0.5]])
    assert macro_auroc(y, p) == 1.0


def test_macro_auroc_nan_when_every_column_collapses() -> None:
    y = np.zeros((4, 2))
    p = np.random.default_rng(0).random((4, 2))
    assert np.isnan(macro_auroc(y, p))


def test_metric_for_dispatches_to_correct_fn() -> None:
    assert metric_for(EvalMetric.rmse) is rmse
    assert metric_for(EvalMetric.macro_auroc) is macro_auroc
    assert metric_for(EvalMetric.auroc) is auroc
