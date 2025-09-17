from dataclasses import dataclass, field
import numpy as np
from typing import Any

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
    train_test_split,
)
from tqdm import tqdm
from threedscriptors.evaluation.regression.learner import Learner
from threedscriptors.evaluation.regression.utils import _aggregate_params


@dataclass
class CVParams:
    n_splits: int = (5,)
    n_repeats: int = (3,)
    random_state: int = (42,)
    scoring: str
    tune: bool
    n_inner_splits = 5
    n_random_search_iterations: int = 5

    @property
    def n_total(self):
        return self.n_splits * self.n_repeats


@dataclass
class CVResult:
    fold_metrics: np.ndarray  # shape: (n_outer_folds, 3) -> [MAE, RMSE, R2]
    best_params_per_fold: list[dict[str, Any]]  # tuned params (or fixed) per fold
    final_params: dict[str, Any]  # aggregated params used for final refit
    info: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        m = np.asarray(self.fold_metrics)
        means = m.mean(axis=0)
        stds = m.std(axis=0, ddof=1 if m.shape[0] > 1 else 0)
        n = m.shape[0]

        params = ", ".join(
            f"{k}={self.final_params.get(k, 'NA')}" for k in self.final_params.keys()
        )

        return ( f"Results from {n}-fold CV:"
            f"MAE: {means[0]:.4f}±{stds[0]:.4f} | "
            f"RMSE: {means[1]:.4f}±{stds[1]:.4f} | "
            f"R²: {means[2]:.4f}±{stds[2]:.4f}\n"
            f"With parameters: \n"
            f"{params}"
        )

    __str__ = __repr__


def run_kfold_repeated_cross_validation(
    X,
    y,
    *,
    cv_params: CVParams,
    learner: Learner,
) -> CVResult:
    """
    Outer: RepeatedKFold for generalization estimate.
    Optional inner: RandomizedSearchCV to tune hyperparams on the training fold.
    Final: refit on ALL data with aggregated best params.
    """

    X_use = np.asarray(X)
    y_use = np.asarray(y).ravel()

    outer = RepeatedKFold(
        n_splits=cv_params.n_splits,
        n_repeats=cv_params.n_repeats,
        random_state=cv_params.random_state,
    )

    fold_metrics: list[list[float]] = []
    best_params_per_fold: list[dict[str, Any]] = []

    fold_id = 0
    for tr_idx, te_idx in tqdm(outer.split(X_use, y_use), total=cv_params.n_total):
        seed = cv_params.random_state + fold_id

        X_tr, X_te = (X_use[tr_idx], X_use[te_idx])
        y_tr, y_te = (y_use[tr_idx], y_use[te_idx])

        if cv_params.tune:
            base = learner.build_estimator(
                learner.params, random_state=cv_params.random_state + fold_id
            )
            inner = KFold(
                n_splits=cv_params.n_inner_splits, shuffle=True, random_state=seed
            )
            rs = RandomizedSearchCV(
                estimator=base,
                param_distributions=learner.search_space,
                n_iter=cv_params.n_random_search_iterations,
                cv=inner,
                scoring=cv_params.scoring,
                n_jobs=-1,
                refit=True,
                random_state=seed,
                verbose=0,
            )
            rs.fit(X_tr, y_tr)
            est = rs.best_estimator_
            best_params = learner.normalize_model_params_for_aggregation(
                rs.best_params_
            )
        else:
            est = learner.build_estimator(random_state=seed)
            est.fit(X_tr, y_tr)
            best_params = {}

        y_hat = est.predict(X_te)

        mae = mean_absolute_error(y_te, y_hat)
        rmse = np.sqrt(mean_squared_error(y_te, y_hat))
        r2 = r2_score(y_te, y_hat)

        fold_metrics.append([mae, rmse, r2])

        best_params_per_fold.append(best_params)
        fold_id += 1

    fold_metrics = np.asarray(fold_metrics)

    # Aggregate params across folds and refit on ALL data
    aggregated_params = _aggregate_params(best_params_per_fold)

    final_params = learner._finalize_params(aggregated_params)

    final_model = learner.build_estimator(final_params)
    final_model.fit(X_use, y_use)

    return CVResult(
        fold_metrics=fold_metrics,
        best_params_per_fold=best_params_per_fold,
        final_params=final_params,
        final_model=final_model,
        info={"cross-validation_params": cv_params, "learner": learner.name},
    )
