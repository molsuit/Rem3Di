
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
)
from sklearn.pipeline import Pipeline





def ridge_repeated_kfold_cv(
    X,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    inner_splits: int = 5,
    scaler = "passthrough",
    alphas: Sequence[float] | None = None,
    random_state: int = 42,
    scoring: str = "neg_mean_absolute_error",
    n_jobs: int = -1,
) -> RidgeCVResult:
    """
    Train/evaluate Ridge with nested CV:
      - Outer loop: RepeatedKFold for generalization estimate
      - Inner loop: KFold GridSearchCV over alpha
      - Returns per-fold metrics, best alphas, and a final refit model using
        median(alpha_best) on the full dataset.

    Parameters
    ----------
    X : array-like or scipy.sparse matrix, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    n_splits : outer CV splits
    n_repeats : outer CV repeats
    inner_splits : inner CV splits for alpha selection
    alphas : grid of ridge alphas; defaults to logspace(-6, 6, 25)
    scoring : GridSearchCV scoring (default MAE)
    n_jobs : parallelism for GridSearchCV

    Returns
    -------
    RidgeCVResult
    """
    X = X  # allow dense or sparse
    y = np.asarray(y)

    if alphas is None:
        alphas = np.logspace(-6, 6, 25)

    outer_cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    metrics = []     # list of [mae, rmse, r2]
    best_as = []     # best alpha per outer fold

    # Define a pipeline so scaling happens inside each CV split (no leakage)
    pipe = Pipeline([
        ("scaler", scaler),
        ("ridge", Ridge(fit_intercept=True, random_state=None))  # Ridge has no RNG in fitting
    ])
    param_grid = {"ridge__alpha": alphas}

    fold_id = 0

    for train_idx, test_idx in outer_cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state + fold_id)

        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
        )
        grid.fit(X_tr, y_tr)
        best_alpha = grid.best_params_["ridge__alpha"]

        # Evaluate on held-out test
        y_pred = grid.best_estimator_.predict(X_te)
        mae = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2 = r2_score(y_te, y_pred)

        metrics.append([mae, rmse, r2])
        best_as.append(best_alpha)
        fold_id += 1

    metrics = np.asarray(metrics)
    best_as = np.asarray(best_as)
    final_alpha = float(np.median(best_as))

    # Refit final model on all data using the robust median alpha
    final_pipe = Pipeline([
        ("scaler", scaler),
        ("ridge", Ridge(alpha=final_alpha, fit_intercept=True))
    ])
    final_pipe.fit(X, y)

    return RidgeCVResult(
        fold_metrics=metrics,
        fold_alphas=best_as,
        final_alpha=final_alpha,
        final_model=final_pipe,
        info={
            "alphas_grid": np.asarray(alphas),
            "scoring": scoring,
            "n_outer_folds": n_splits * n_repeats,
            "inner_splits": inner_splits,
        },
    )
