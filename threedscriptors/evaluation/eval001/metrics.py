from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from scipy.stats import spearmanr


@dataclass(frozen=True)
class MetricResult:
    metric_name: str
    value: float
    n_labels: int
    n_labels_scored: int
    notes: str = ""


def regression_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> MetricResult:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    metric_upper = metric.upper()
    if len(y_true) == 0:
        return MetricResult(metric, float("nan"), 1, 0, "no_valid_pairs")
    if metric_upper == "MAE":
        value = mean_absolute_error(y_true, y_pred)
        metric_name = "MAE"
    elif metric_upper == "RMSE":
        value = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metric_name = "RMSE"
    elif metric_upper == "R2":
        value = r2_score(y_true, y_pred)
        metric_name = "R2"
    elif metric_upper in {"SPEARMAN", "SPEARMANR", "SPEARMAN_R", "SPEARMANRHO"}:
        if len(y_true) < 2 or len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            return MetricResult("Spearman", float("nan"), 1, 0, "constant_or_too_small")
        value = spearmanr(y_true, y_pred).statistic
        metric_name = "Spearman"
    else:
        raise ValueError(f"Unsupported regression metric: {metric}")
    return MetricResult(metric_name, float(value), 1, 1)


def binary_metric(y_true: np.ndarray, y_score: np.ndarray, metric: str = "AUROC") -> MetricResult:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    mask = ~np.isnan(y_true) & ~np.isnan(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(np.unique(y_true)) < 2:
        return MetricResult(metric, float("nan"), 1, 0, "single_class_test")
    if metric.upper() in {"AUROC", "ROC_AUC"}:
        value = roc_auc_score(y_true, y_score)
        metric_name = "AUROC"
    elif metric.upper() in {"AUPRC", "AP", "PR-AUC", "PRAUC", "PR_AUC"}:
        value = average_precision_score(y_true, y_score)
        metric_name = "AUPRC"
    else:
        raise ValueError(f"Unsupported binary metric: {metric}")
    return MetricResult(metric_name, float(value), 1, 1)


def multilabel_macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> MetricResult:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, 1)

    values: list[float] = []
    skipped: list[str] = []
    for j in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, j]) & ~np.isnan(y_score[:, j])
        yt = y_true[mask, j]
        yp = y_score[mask, j]
        if len(yt) == 0:
            skipped.append(f"{j}:no_labels")
            continue
        if len(np.unique(yt)) < 2:
            skipped.append(f"{j}:single_class_test")
            continue
        values.append(float(roc_auc_score(yt, yp)))
    if not values:
        return MetricResult("macro-AUROC", float("nan"), y_true.shape[1], 0, ";".join(skipped))
    return MetricResult(
        "macro-AUROC",
        float(np.mean(values)),
        y_true.shape[1],
        len(values),
        ";".join(skipped),
    )
