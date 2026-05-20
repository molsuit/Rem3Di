"""Metric functions for the descriptor-probe benchmark.

One entry per :class:`EvalMetric`. Each function takes ``y_true``, ``y_pred``
and returns a float; the caller picks the right one via :func:`metric_for`.
Multilabel inputs (``ClinTox``/``SIDER``/``Tox21``) come in as ``(N, K)``;
per-column metrics are averaged with NaN-column skipping so single-class test
folds don't crash the panel.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from threedscriptors.data_handling.benchmarks import EvalMetric


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


def auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_pred))


def auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_pred))


def macro_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Per-column AUROC averaged across multilabel targets.

    Skips columns whose test fold collapses to a single class — sklearn would
    raise on those, and they carry no AUROC signal anyway. Returns NaN when
    no column is scorable.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim != 2:
        yt = yt.reshape(-1, 1)
        yp = yp.reshape(-1, 1)
    per_col: list[float] = []
    for j in range(yt.shape[1]):
        col_t = yt[:, j]
        col_p = yp[:, j]
        keep = ~np.isnan(col_t)
        col_t = col_t[keep]
        col_p = col_p[keep]
        if len(np.unique(col_t)) < 2:
            continue
        per_col.append(float(roc_auc_score(col_t, col_p)))
    return float(np.mean(per_col)) if per_col else float("nan")


_METRIC_FNS: dict[EvalMetric, Callable[[np.ndarray, np.ndarray], float]] = {
    EvalMetric.rmse: rmse,
    EvalMetric.mae: mae,
    EvalMetric.r2: r2,
    EvalMetric.spearman: spearman,
    EvalMetric.auroc: auroc,
    EvalMetric.auprc: auprc,
    EvalMetric.macro_auroc: macro_auroc,
}


def metric_for(metric: EvalMetric) -> Callable[[np.ndarray, np.ndarray], float]:
    return _METRIC_FNS[metric]
