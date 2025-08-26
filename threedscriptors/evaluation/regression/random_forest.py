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


# -------------------------
# Trainer
# -------------------------

def rf_repeated_kfold_cv(
    X,
    y,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    rf_params: RFParams | None = None,
    tune: bool = False,
    param_distributions: dict[str, Sequence[Any]] | None = None,
    n_iter: int = 25,
    inner_splits: int = 5,
    scoring: str = "neg_mean_absolute_error",
    n_jobs_search: int | None = None,
) -> RFCVResult:
    """
    Outer: RepeatedKFold for generalization estimate.
    Optional inner: RandomizedSearchCV to tune hyperparams on the training fold.
    Final: refit on ALL data with aggregated best params.
    """



    X_use = np.asarray(X)
    y_use = np.asarray(y).ravel()

    if rf_params is None:
        rf_params = RFParams(random_state=random_state)

    if tune and param_distributions is None:
        param_distributions = default_rf_param_distributions()

    outer = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_metrics: list[list[float]] = []
    best_params_per_fold: list[dict[str, Any]] = []

    fold_id = 0
    for tr_idx, te_idx in tqdm(outer.split(X_use, y_use), total = n_splits * n_repeats):
        X_tr, X_te = (X_use[tr_idx], X_use[te_idx])
        y_tr, y_te = (y_use[tr_idx], y_use[te_idx])

        if tune:
            base = RandomForestRegressor(**rf_params.to_dict(), random_state=random_state + fold_id)
            inner = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state + 777 + fold_id)
            rs = RandomizedSearchCV(
                estimator=base,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=inner,
                scoring=scoring,
                n_jobs=(n_jobs_search if n_jobs_search is not None else rf_params.n_jobs),
                refit=True,
                random_state=random_state + fold_id,
                verbose=0,
            )
            rs.fit(X_tr, y_tr)
            best_params = rs.best_params_
        else:
            best_params = {}

        fold_params = {**rf_params.to_dict(), **best_params, "random_state": random_state + fold_id}


        model = RandomForestRegressor(**fold_params)
        model.fit(X_tr, y_tr)

        y_hat = model.predict(X_te)
        mae = mean_absolute_error(y_te, y_hat)
        rmse = np.sqrt(mean_squared_error(y_te, y_hat))
        r2 = r2_score(y_te, y_hat)

        fold_metrics.append([mae, rmse, r2])
        best_params_per_fold.append({**rf_params.to_dict(), **best_params})
        fold_id += 1

    fold_metrics = np.asarray(fold_metrics)

    # Aggregate params across folds and refit on ALL data
    aggregated = _aggregate_params(best_params_per_fold)
    final_params = {**rf_params.to_dict(), **aggregated, "random_state": random_state + 9999}
    for k in ("bootstrap", "oob_score"):
        if k in final_params:
            final_params[k] = bool(final_params[k])
    final_model = RandomForestRegressor(**final_params)
    final_model.fit(X_use, y_use)

    return RFCVResult(
        fold_metrics=fold_metrics,
        best_params_per_fold=best_params_per_fold,
        final_params=final_params,
        final_model=final_model,
        info={
            "n_outer_folds": n_splits * n_repeats,
            "tuned": tune,
            "inner_splits": inner_splits if tune else 0,
            "param_distributions": param_distributions if tune else None,
            "scoring": scoring,
        },
    )
