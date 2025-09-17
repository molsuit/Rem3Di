from collections.abc import Mapping
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



def _aggregate_params(best_params_list: list[dict]) -> dict:
    if not best_params_list:
        return {}

    keys = sorted({k for d in best_params_list for k in d.keys()})
    agg = {}
    for k in keys:
        vals = [d[k] for d in best_params_list if k in d]
        # drop None to avoid poisoning medians/modes (e.g., max_samples when bootstrap=False)
        vals = [v for v in vals if v is not None]
        if not vals:
            continue

        v0 = vals[0]
        # IMPORTANT: bool BEFORE int (bool is a subclass of int)
        if isinstance(v0, (bool, np.bool_)):
            true_frac = np.mean([bool(v) for v in vals])
            agg[k] = bool(true_frac >= 0.5)   # majority vote
        elif isinstance(v0, (float, np.floating)):
            agg[k] = float(np.median(vals))
        elif isinstance(v0, (int, np.integer)):
            agg[k] = int(np.median(vals))
        elif isinstance(v0, str):
            uniq, counts = np.unique(vals, return_counts=True)
            agg[k] = uniq[np.argmax(counts)]
        else:
            # fallback: first value
            agg[k] = v0

    return agg

