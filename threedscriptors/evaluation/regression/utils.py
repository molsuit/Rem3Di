from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def _cv_means_stds(fold_metrics: np.ndarray):
    m = np.asarray(fold_metrics, dtype=float)
    means = m.mean(axis=0)
    stds  = m.std(axis=0, ddof=1 if m.shape[0] > 1 else 0)
    return means, stds  # order: [MAE, RMSE, R2]

def cv_results_to_nested_dict(
    task_name: str,
    model_to_cvresult: Mapping[str, Any],   # values: RidgeCVResult or RFCVResult
    *,
    round_to: int | None = None,
    include_std: bool = False,
    include_folds: bool = False,
    include_meta: bool = False,             # add alpha/params if available
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Convert a mapping {model_name: CVResult} into {task: {model: {MAE,MSE,R2,(...)} } }.
    Works for RidgeCVResult and RFCVResult since both expose `fold_metrics`.
    """
    task_block: dict[str, dict[str, Any]] = {}

    for model_name, res in model_to_cvresult.items():
        means, stds = _cv_means_stds(res.fold_metrics)
        mae, rmse, r2 = (float(means[0]), float(means[1]), float(means[2]))
        if round_to is not None:
            mae = round(mae, round_to); rmse = round(rmse, round_to); r2 = round(r2, round_to)

        entry: dict[str, Any] = {"MAE": mae, "MSE": rmse, "R2": r2}
        task_block[str(model_name)] = entry
    return {str(task_name) : task_block}


# ---------- Utilities ----------
def dc_to_dict(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj): return asdict(obj)
    if isinstance(obj, dict): return dict(obj)
    raise TypeError("Expected a dataclass or dict.")

def prefix(step: str, d: dict[str, Any]) -> dict[str, Any]:
    return {f"{step}__{k}": v for k, v in d.items()}

def unprefix(step: str, d: dict[str, Any]) -> dict[str, Any]:
    p = f"{step}__"
    return {k[len(p):]: v for k, v in d.items() if k.startswith(p)}

def is_bool(x) -> bool:
    return isinstance(x, (bool, np.bool_))

def is_numeric_ex_bool(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not is_bool(x)

def mode(values):
    counts, order = {}, {}
    for i, v in enumerate(values):
        counts[v] = counts.get(v, 0) + 1
        order.setdefault(v, i)
    return sorted(counts.items(), key=lambda kv: (-kv[1], order[kv[0]]))[0][0]
