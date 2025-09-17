from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, RepeatedKFold
from tqdm import tqdm

# -------------------------
# Dataclasses
# -------------------------

@dataclass
class RFParams:
    n_estimators: int = 600
    max_depth: int | None = None
    max_features: Any = "sqrt"      # {"auto","sqrt","log2"} or float in (0,1]
    min_samples_split: int = 2
    min_samples_leaf: int = 2
    bootstrap: bool = True
    max_samples: float | None = 0.8  # only used if bootstrap=True; float in (0,1]
    oob_score: bool = False
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "bootstrap": self.bootstrap,
            "max_samples": self.max_samples if self.bootstrap else None,
            "oob_score": self.oob_score if self.bootstrap else False,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
        }
        return {k: v for k, v in d.items() if v is not None}


    def default_rf_param_distributions() -> dict[str, Sequence[Any]]:
        return {
        "n_estimators":       [200, 400, 800, 1200],
        "max_depth":          [None, 8, 12, 16],
        "max_features":       ["sqrt", 0.3, 0.5, 0.8],
        "min_samples_leaf":   [1, 2, 4, 8],
        "min_samples_split":  [2, 5, 10],
        "bootstrap":          [True],            # keep True to allow max_samples, oob
        "max_samples":        [0.5, 0.7, 0.9],
    }


@dataclass
class RFCVResult:
    fold_metrics: np.ndarray                     # (n_outer_folds, 3) -> [MAE, RMSE, R2]
    best_params_per_fold: list[dict[str, Any]]
    final_params: dict[str, Any]
    final_model: RandomForestRegressor
    info: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        m = np.asarray(self.fold_metrics)
        if m.ndim != 2 or m.shape[1] < 3 or m.size == 0:
            return "RandomForest Repeated CV: no metrics."
        means = m.mean(axis=0)
        stds = m.std(axis=0, ddof=1 if m.shape[0] > 1 else 0)
        n = m.shape[0]
        keys = ["n_estimators", "max_depth", "max_features", "min_samples_leaf",
                "min_samples_split", "bootstrap", "max_samples"]
        shown = ", ".join(f"{k}={self.final_params.get(k, 'NA')}" for k in keys)
        return (f"RandomForest Repeated CV (n={n}) | "
                f"MAE: {means[0]:.4f}±{stds[0]:.4f} | "
                f"RMSE: {means[1]:.4f}±{stds[1]:.4f} | "
                f"R²: {means[2]:.4f}±{stds[2]:.4f}\n"
                f"Final params: {shown}")

    __str__ = __repr__


# -------------------------
# Search space + aggregation
# -------------------------




# -------------------------
# Trainer
# -------------------------
