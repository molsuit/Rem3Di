from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import RepeatedKFold, KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import lightgbm as lgb  # for callbacks. If you prefer, you can omit callbacks entirely.
from tqdm import tqdm




@dataclass
class LGBMParams:
    # Core
    n_estimators: int = 4000
    learning_rate: float = 0.03
    num_leaves: int = 256
    max_depth: int = -1
    min_child_samples: int = 20  # a.k.a. min_data_in_leaf
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    # Misc
    boosting_type: str = "gbdt"
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = -1
    importance_type: str = "gain"  # for feature_importances_

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "boosting_type": self.boosting_type,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": self.verbosity,
            "importance_type": self.importance_type,
        }
        # Drop None values
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class LGBMCVResult:
    fold_metrics: np.ndarray                     # shape: (n_outer_folds, 3) -> [MAE, RMSE, R2]
    best_params_per_fold: List[Dict[str, Any]]   # tuned params (or fixed) per fold
    final_params: Dict[str, Any]                 # aggregated params used for final refit
    final_model: LGBMRegressor                   # fitted on all data with final_params
    info: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        m = np.asarray(self.fold_metrics)
        if m.ndim != 2 or m.shape[1] < 3 or m.size == 0:
            return "LightGBM CV: no metrics."

        means = m.mean(axis=0)
        stds = m.std(axis=0, ddof=1 if m.shape[0] > 1 else 0)
        n = m.shape[0]

        # Compact pretty printing of final params (a few key ones)
        keys = ["learning_rate", "num_leaves", "min_child_samples", "subsample",
                "colsample_bytree", "reg_lambda", "reg_alpha", "n_estimators", "max_depth"]
        shown = ", ".join(f"{k}={self.final_params.get(k, 'NA')}" for k in keys)

        return (
            f"LightGBM Repeated CV (n={n}) | "
            f"MAE: {means[0]:.4f}±{stds[0]:.4f} | "
            f"RMSE: {means[1]:.4f}±{stds[1]:.4f} | "
            f"R²: {means[2]:.4f}±{stds[2]:.4f}\n"
            f"Final params: {shown}"
        )

    __str__ = __repr__


def default_lgbm_param_distributions() -> Dict[str, Sequence[Any]]:
    """
    Reasonable, compact distributions for RandomizedSearchCV (discrete choices; no SciPy required).
    """
    return {
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
        "num_leaves": [31, 63, 127, 255, 511],
        "max_depth": [-1, 8, 12, 16],
        "min_child_samples": [5, 10, 20, 40, 80],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 0.1, 1.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "n_estimators": [1500, 3000, 6000, 9000],
        # You can also vary boosting_type: ["gbdt", "goss"] if desired.
    }



# -------------------------
# Aggregation utility
# -------------------------

def _aggregate_params(best_params_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-fold best params into a single robust set:
    - numeric -> median
    - integers -> median then round
    - categorical (str) -> mode
    """
    if not best_params_list:
        return {}

    keys = sorted({k for d in best_params_list for k in d.keys()})
    agg: Dict[str, Any] = {}
    for k in keys:
        vals = [d[k] for d in best_params_list if k in d]
        if not vals:
            continue
        if isinstance(vals[0], (int, np.integer)):
            agg[k] = int(np.median(vals))
        elif isinstance(vals[0], (float, np.floating)):
            agg[k] = float(np.median(vals))
        elif isinstance(vals[0], str):
            # mode
            uniq, counts = np.unique(vals, return_counts=True)
            agg[k] = uniq[np.argmax(counts)]
        else:
            # fallback to first
            agg[k] = vals[0]
    return agg


# -------------------------
# Trainer with Repeated K-Fold and optional tuning
# -------------------------

def lightgbm_repeated_kfold_cv(
    X,
    y,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    lgbm_params: Optional[LGBMParams] = None,
    tune: bool = False,
    param_distributions: Optional[Dict[str, Sequence[Any]]] = None,
    n_iter: int = 25,
    inner_splits: int = 5,
    scoring: str = "neg_mean_absolute_error",
    use_early_stopping: bool = True,
    early_stopping_rounds: int = 100,
    val_size: float = 0.1,
    n_jobs_search: Optional[int] = None,
) -> LGBMCVResult:
    """
    Outer: RepeatedKFold for generalization estimate.
    Optional inner: RandomizedSearchCV to tune hyperparameters on training fold.
    Final: refit on ALL data with aggregated best params (median/mode) and internal val split for early stopping.

    Notes:
      - No scaling is needed for LightGBM.
      - Early stopping uses a small validation split carved from the training data of each outer fold (no leakage).
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if lgbm_params is None:
        lgbm_params = LGBMParams(random_state=random_state)

    if tune and param_distributions is None:
        param_distributions = default_lgbm_param_distributions()

    outer = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_metrics: List[List[float]] = []
    best_params_per_fold: List[Dict[str, Any]] = []

    fold_id = 0
    for tr_idx, te_idx in tqdm(outer.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Optional hyperparameter tuning (nested CV) on training portion
        if tune:
            base_est = LGBMRegressor(**lgbm_params.to_dict())
            inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state + fold_id)
            rs = RandomizedSearchCV(
                estimator=base_est,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=inner_cv,
                scoring=scoring,
                n_jobs=(n_jobs_search if n_jobs_search is not None else lgbm_params.n_jobs),
                refit=True,
                verbose=0,
                random_state=random_state + fold_id,
            )
            rs.fit(X_tr, y_tr)  # NOTE: early stopping is not used inside RS CV
            best_params = rs.best_params_
        else:
            best_params = {}

        # Build model for this fold using tuned (or fixed) params
        fold_params = {**lgbm_params.to_dict(), **best_params}

        # For early stopping, split training into train/val (no leakage)
        if use_early_stopping and early_stopping_rounds > 0:
            X_tr_in, X_val_in, y_tr_in, y_val_in = train_test_split(
                X_tr, y_tr, test_size=val_size, random_state=random_state + 1234 + fold_id
            )
            model = LGBMRegressor(**fold_params)
            model.fit(
                X_tr_in, y_tr_in,
                eval_set=[(X_val_in, y_val_in)],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
        else:
            model = LGBMRegressor(**fold_params)
            model.fit(X_tr, y_tr)

        # Evaluate on held-out test fold
        y_hat = model.predict(X_te)
        mae = mean_absolute_error(y_te, y_hat)
        rmse = np.sqrt(mean_squared_error(y_te, y_hat))
        r2 = r2_score(y_te, y_hat)

        fold_metrics.append([mae, rmse, r2])
        best_params_per_fold.append(best_params if tune else lgbm_params.to_dict())
        fold_id += 1

    fold_metrics = np.asarray(fold_metrics)

    # Aggregate best params across folds; merge over the base defaults
    aggregated = _aggregate_params(best_params_per_fold)
    final_params = {**lgbm_params.to_dict(), **aggregated}

    # Final refit on ALL data (use an internal val split for early stopping)
    if use_early_stopping and early_stopping_rounds > 0:
        X_all_tr, X_all_val, y_all_tr, y_all_val = train_test_split(
            X, y, test_size=max(val_size, 0.1), random_state=random_state + 9999
        )
        final_model = LGBMRegressor(**final_params)
        final_model.fit(
            X_all_tr, y_all_tr,
            eval_set=[(X_all_val, y_all_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
    else:
        final_model = LGBMRegressor(**final_params)
        final_model.fit(X, y)

    return LGBMCVResult(
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