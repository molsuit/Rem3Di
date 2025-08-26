from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional
import numpy as np
import yaml



def _cv_means_stds(fold_metrics: np.ndarray):
    m = np.asarray(fold_metrics, dtype=float)
    means = m.mean(axis=0)
    stds  = m.std(axis=0, ddof=1 if m.shape[0] > 1 else 0)
    return means, stds  # order: [MAE, RMSE, R2]

def cv_results_to_nested_dict(
    task_name: str,
    model_to_cvresult: Mapping[str, Any],   # values: RidgeCVResult or RFCVResult
    *,
    round_to: Optional[int] = None,
    include_std: bool = False,
    include_folds: bool = False,
    include_meta: bool = False,             # add alpha/params if available
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Convert a mapping {model_name: CVResult} into {task: {model: {MAE,MSE,R2,(...)} } }.
    Works for RidgeCVResult and RFCVResult since both expose `fold_metrics`.
    """
    task_block: Dict[str, Dict[str, Any]] = {}

    for model_name, res in model_to_cvresult.items():
        means, stds = _cv_means_stds(res.fold_metrics)
        mae, rmse, r2 = (float(means[0]), float(means[1]), float(means[2]))
        if round_to is not None:
            mae = round(mae, round_to); rmse = round(rmse, round_to); r2 = round(r2, round_to)

        entry: Dict[str, Any] = {"MAE": mae, "MSE": rmse, "R2": r2}
        task_block[str(model_name)] = entry
    return {str(task_name) : task_block}