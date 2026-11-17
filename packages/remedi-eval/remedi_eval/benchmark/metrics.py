"""Metric functions for the descriptor-probe benchmark.

One entry per :class:`EvalMetric`. Each function takes ``y_true``, ``y_pred``
and returns a float; the caller picks the right one via :func:`metric_for`.
Multilabel inputs (``ClinTox``/``SIDER``/``Tox21``) come in as ``(N, K)``;
per-column metrics are averaged with NaN-column skipping so single-class test
folds don't crash the panel.

:func:`multiclass_report` is the one non-scalar entry: the per-class table and
confusion matrix that every multiclass cell emits alongside its headline
number. It replaced the chirality-specific per-class report task (§6), so every
future multiclass benchmark gets per-class reporting for free.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

from remedi_data.format import EvalMetric


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


def auroc_one_vs_rest_per_class(
    y_true: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """One-vs-rest AUROC of every class column of ``probabilities``.

    Class ``c`` is scored as the binary problem "is class ``c``" on column
    ``c``. A class with no positive or no negative in ``y_true`` (absent from
    the fold, or the only class in it) has no AUROC and gets NaN, as does every
    class when a prediction is non-finite (e.g. a diverged model).
    """
    true_labels = np.asarray(y_true).reshape(-1).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    n_classes = probabilities.shape[1]
    per_class = np.full(n_classes, np.nan)
    if not np.isfinite(probabilities).all():
        return per_class
    for class_index in range(n_classes):
        is_class = true_labels == class_index
        if 0 < is_class.sum() < len(is_class):
            per_class[class_index] = roc_auc_score(
                is_class, probabilities[:, class_index]
            )
    return per_class


def macro_auroc_ovr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """One-vs-rest macro AUROC from the ``(n, n_classes)`` probability matrix.

    The mean of :func:`auroc_one_vs_rest_per_class` over the classes that can
    be scored, i.e. that have both positives and negatives in the fold; a
    declared class absent from the test fold is skipped, like a single-class
    column in :func:`macro_auroc`. NaN when no class is scorable (fewer than
    two classes in the fold) or a prediction is non-finite. The per-class
    values, and so which classes were scored, are in the ``auroc_ovr`` column
    of :func:`multiclass_report`.
    """
    probabilities = np.asarray(y_pred)
    if probabilities.ndim != 2:
        return float("nan")
    per_class = auroc_one_vs_rest_per_class(y_true, probabilities)
    scored = per_class[~np.isnan(per_class)]
    return float(scored.mean()) if scored.size else float("nan")


@dataclass(frozen=True)
class MulticlassReport:
    """The per-class picture a single balanced-accuracy number hides.

    ``per_class`` is one row per class in label order, with the columns
    ``class_id``, ``class_name``, ``precision``, ``recall``, ``f1``,
    ``support`` and ``auroc_ovr`` (the one-vs-rest AUROC; NaN for a class
    without positives or negatives in the fold, or when only labels were
    given). ``confusion_matrix`` is the integer ``(n_classes, n_classes)``
    matrix, rows indexed by true class and columns by predicted class, over the
    fixed label order ``0 … n_classes - 1`` — so a class absent from the test
    fold keeps its row and column instead of shifting the others.
    """

    per_class: pd.DataFrame
    confusion_matrix: np.ndarray
    class_names: list[str]


def class_display_names(
    n_classes: int, class_names: Sequence[str] | None = None
) -> list[str]:
    """Display names for classes ``0 … n_classes - 1``.

    ``class_names`` comes from the bundle's ``LabelColumn.class_names``, which
    the spec already validates to have exactly ``n_classes`` entries; anything
    else (including ``None``) falls back to ``class_0 … class_{n-1}``.
    """
    if class_names is not None and len(class_names) == n_classes:
        return [str(name) for name in class_names]
    return [f"class_{index}" for index in range(n_classes)]


def multiclass_report(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_classes: int,
    class_names: Sequence[str] | None = None,
) -> MulticlassReport:
    """Per-class precision / recall / F1 / support plus the confusion matrix.

    Args:
        y_true: ``(n,)`` integer class indices of the scored fold.
        probabilities: ``(n, n_classes)`` class probabilities; the predicted
            label is the argmax. A 1-D array is taken to be labels already.
        n_classes: the declared class count, which pins the label order.
        class_names: display names, one per class; see
            :func:`class_display_names` for the fallback.

    Returns:
        The :class:`MulticlassReport` for this fold. Classes with no predicted
        and no true instance score 0 rather than raising (``zero_division=0``).
    """
    true_labels, predicted_labels = _multiclass_labels(y_true, probabilities)
    labels = np.arange(n_classes)
    names = class_display_names(n_classes, class_names)

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=labels, zero_division=0
    )
    probabilities = np.asarray(probabilities)
    auroc_per_class = (
        auroc_one_vs_rest_per_class(true_labels, probabilities)
        if probabilities.ndim == 2
        else np.full(n_classes, np.nan)
    )
    per_class = pd.DataFrame(
        {
            "class_id": labels,
            "class_name": names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "auroc_ovr": auroc_per_class,
        }
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    return MulticlassReport(
        per_class=per_class, confusion_matrix=matrix, class_names=names
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


# ``EvalMetric`` lives in remedi_data, which knows nothing of how a metric is
# read; the direction is an evaluation concern, so it is kept here.
_HIGHER_IS_BETTER: dict[EvalMetric, bool] = {
    EvalMetric.rmse: False,
    EvalMetric.mae: False,
    EvalMetric.r2: True,
    EvalMetric.spearman: True,
    EvalMetric.auroc: True,
    EvalMetric.auprc: True,
    EvalMetric.macro_auroc: True,
    EvalMetric.balanced_accuracy: True,
    EvalMetric.macro_f1: True,
    EvalMetric.macro_auroc_ovr: True,
    EvalMetric.pair_ranking_accuracy: True,
}


def higher_is_better(metric: EvalMetric) -> bool:
    """Whether a larger value of ``metric`` is a better model."""
    return _HIGHER_IS_BETTER[metric]
