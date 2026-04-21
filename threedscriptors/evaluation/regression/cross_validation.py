import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
)
from tqdm import tqdm

from threedscriptors.evaluation.regression.learner import Learner
from threedscriptors.evaluation.regression.utils import dc_to_dict
from threedscriptors.evaluation.results import PydanticResult

# Silence linalg warnings from NumPy & SciPy
warnings.filterwarnings("ignore", category=scipy.linalg.LinAlgWarning)


class CrossValidationOutput(BaseModel):
    """
    Cross-validation result payload to be wrapped by PydanticResult(obj=...).
    """

    # payload
    fold_metrics: np.ndarray  # (n_folds, 3) -> [MAE, RMSE, R2]
    best_params_per_fold: list[dict[str, Any]]
    final_params: dict[str, Any]
    info: dict[str, Any] = Field(default_factory=dict)

    # in-memory only; excluded from dumps
    final_model: BaseEstimator | None = Field(default=None, exclude=True, repr=False)

    # allow numpy/sklearn while we control serialization
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -------- validation / coercion --------
    @field_validator("fold_metrics", mode="before")
    @classmethod
    def _coerce_fold_metrics(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                "fold_metrics must have shape (n_folds, 3) for [MAE, RMSE, R2]."
            )
        return arr

    @staticmethod
    def _jsonable(x: Any) -> Any:
        # Make nested dict/list contents JSON-safe (NumPy scalars → Python)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, dict):
            return {k: CrossValidationOutput._jsonable(v) for k, v in x.items()}
        if isinstance(x, list | tuple):
            return [CrossValidationOutput._jsonable(v) for v in x]
        return x

    @field_validator("best_params_per_fold", "final_params", "info", mode="before")
    @classmethod
    def _normalize_dicts(cls, v: Any) -> Any:
        return cls._jsonable(v)

    # -------- serializers --------
    @field_serializer("fold_metrics")
    def _fold_metrics_to_list(self, v: np.ndarray) -> list:
        return v.tolist()

    # -------- pretty repr --------
    def __repr__(self) -> str:
        m = self.fold_metrics
        n = int(m.shape[0])
        ddof = 1 if n > 1 else 0
        means = m.mean(axis=0)
        stds = m.std(axis=0, ddof=ddof)
        params = ", ".join(
            f"{k}={self.final_params.get(k, 'NA')}" for k in self.final_params.keys()
        )
        return (
            f"Results from {n}-fold CV: "
            f"MAE: {means[0]:.4f}±{stds[0]:.4f} | "
            f"RMSE: {means[1]:.4f}±{stds[1]:.4f} | "
            f"R²: {means[2]:.4f}±{stds[2]:.4f}\n"
            f"With parameters:\n{params}"
        )

    __str__ = __repr__


@dataclass
class CVParams:
    n_splits: int = 5
    n_repeats: int = 3
    random_state: int = 0
    scoring: str = "neg_mean_absolute_error"
    tune: bool = True
    n_inner_splits = 3
    n_random_search_iterations: int = 5
    n_jobs: int = -1

    @property
    def total_folds(self):
        return self.n_splits * self.n_repeats


class CrossValidationRunner:
    def __init__(
        self,
        cv_params: CVParams,  # your CVParams dataclass
        learner: Learner,
    ):
        self.cv_params = cv_params
        self.learner = learner

    def run_kfold_repeated_cross_validation(self, X, y) -> CrossValidationOutput:
        """
        Outer: RepeatedKFold for generalization estimate.
        Optional inner: RandomizedSearchCV to tune hyperparams on the training fold.
        Final: refit on ALL data with aggregated best params.
        """

        # --- Input as arrays ---
        X_use = np.asarray(X)
        y_use = np.asarray(y).ravel()

        # --- Outer CV ---
        outer = RepeatedKFold(
            n_splits=self.cv_params.n_splits,
            n_repeats=self.cv_params.n_repeats,
            random_state=self.cv_params.random_state,
        )

        fold_metrics: list[list[float]] = []
        best_params_per_fold: list[dict[str, Any]] = []

        # --- Iterate folds ---
        for fold_id, (tr_idx, te_idx) in enumerate(
            tqdm(outer.split(X_use, y_use), total=self.cv_params.total_folds)
        ):
            seed = self.cv_params.random_state + fold_id
            X_tr, X_te = (X_use[tr_idx], X_use[te_idx])
            y_tr, y_te = (y_use[tr_idx], y_use[te_idx])

            if self.cv_params.tune:
                base = self.learner.build_estimator(random_state=seed)

                inner = KFold(
                    n_splits=self.cv_params.n_inner_splits,
                    shuffle=True,
                    random_state=seed,
                )

                rs = RandomizedSearchCV(
                    estimator=base,
                    param_distributions=self.learner.get_prefixed_search_space(),  # <- prefixed
                    n_iter=self.cv_params.n_random_search_iterations,
                    cv=inner,
                    scoring=self.cv_params.scoring,
                    n_jobs=self.cv_params.n_jobs,
                    refit=True,
                    random_state=seed,
                    verbose=0,
                )
                rs.fit(X_tr, y_tr)
                est = rs.best_estimator_

                best_params = self.learner.normalize_model_params_for_aggregation(
                    rs.best_params_
                )  # <- unprefix
            else:
                est = self.learner.build_estimator(random_state=seed)
                est.fit(X_tr, y_tr)
                best_params = {}

            # Evaluate on the held-out fold
            y_hat = est.predict(X_te)
            mae = mean_absolute_error(y_te, y_hat)
            rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))
            r2 = r2_score(y_te, y_hat)

            fold_metrics.append([mae, rmse, r2])
            best_params_per_fold.append(best_params)

        fold_metrics = np.asarray(fold_metrics, dtype=float)

        # --- Aggregate params across folds and refit on ALL data ---
        aggregated_params = self.learner._aggregate_params(best_params_per_fold)
        final_params = self.learner._finalize_params(
            aggregated_params
        )  # merge onto defaults & cast

        final_model = self.learner.build_estimator(params=final_params)
        final_model.fit(X_use, y_use)

        self.result = CrossValidationOutput(
            fold_metrics=fold_metrics,
            best_params_per_fold=best_params_per_fold,
            final_params=final_params,
            final_model=final_model,
            info={
                "cross_validation_params": dc_to_dict(self.cv_params),
                "learner": self.learner.name,
                "n_folds": self.cv_params.total_folds,
                "scoring": self.cv_params.scoring,
            },
        )

        return self.result

    def write_output(self, output_directory: Path):
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        res = PydanticResult(
            file_name=f"cross_validation_result_{self.learner.name}", obj=self.result
        )
        res.serialize_to(directory=output_directory)
