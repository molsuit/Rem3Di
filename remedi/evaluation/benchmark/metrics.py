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
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from remedi.data_handling.benchmarks import EvalMetric


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


def _multiclass_labels(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce a multiclass ``(y_true (n,), y_pred (n, n_classes))`` pair to
    integer ``(true, argmax-pred)`` label arrays for label-based metrics."""
    yt = np.asarray(y_true).reshape(-1).astype(int)
    yp = np.asarray(y_pred)
    pred_labels = yp.argmax(axis=1) if yp.ndim == 2 else yp.reshape(-1).astype(int)
    return yt, pred_labels


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-class recall (chance = 1/n_classes). Robust to imbalance —
    the headline for the chiral_cat panel where achiral/central dominate."""
    yt, pred = _multiclass_labels(y_true, y_pred)
    return float(balanced_accuracy_score(yt, pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean per-class F1 over the multiclass argmax prediction."""
    yt, pred = _multiclass_labels(y_true, y_pred)
    return float(f1_score(yt, pred, average="macro", zero_division=0))


def macro_auroc_ovr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """One-vs-rest macro AUROC from the ``(n, n_classes)`` probability matrix.

    Returns NaN when fewer than two distinct classes appear in the test fold
    (AUROC is undefined). ``labels`` pins the class ordering to the probability
    columns so a class missing from the test fold doesn't shift the columns.
    """
    yt = np.asarray(y_true).reshape(-1).astype(int)
    yp = np.asarray(y_pred)
    # Undefined with <2 classes; and never let non-finite predictions (e.g. a
    # diverged model) raise inside roc_auc_score and sink the whole run.
    if yp.ndim != 2 or len(np.unique(yt)) < 2 or not np.isfinite(yp).all():
        return float("nan")
    return float(
        roc_auc_score(
            yt,
            yp,
            multi_class="ovr",
            average="macro",
            labels=np.arange(yp.shape[1]),
        )
    )


_METRIC_FNS: dict[EvalMetric, Callable[[np.ndarray, np.ndarray], float]] = {
    EvalMetric.rmse: rmse,
    EvalMetric.mae: mae,
    EvalMetric.r2: r2,
    EvalMetric.spearman: spearman,
    EvalMetric.auroc: auroc,
    EvalMetric.auprc: auprc,
    EvalMetric.macro_auroc: macro_auroc,
    EvalMetric.balanced_accuracy: balanced_accuracy,
    EvalMetric.macro_f1: macro_f1,
    EvalMetric.macro_auroc_ovr: macro_auroc_ovr,
}


def metric_for(metric: EvalMetric) -> Callable[[np.ndarray, np.ndarray], float]:
    return _METRIC_FNS[metric]
