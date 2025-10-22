from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from threedscriptors.evaluation.regression.utils import (
    dc_to_dict,
    is_bool,
    is_numeric_ex_bool,
    mode,
    prefix,
    unprefix,
)


class ScalerType(Enum):
    NONE = "none"
    STANDARD = "standard"
    MINMAX = "minmax"


@dataclass(frozen=True)
class ScalerParams:
    scaler_type: ScalerType = ScalerType.NONE
    # StandardScaler options
    with_mean: bool = True
    with_std: bool = True
    # MinMaxScaler option
    feature_range: tuple[float, float] | None = None


# ---------- Scaler ----------
class NoOpTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


# ---------- Base Learner (no scaler search space) ----------
P = TypeVar("P")  # model params dataclass
S = TypeVar("S")  # model search-space dataclass


class Learner(ABC, Generic[P, S]):
    name: str
    scaler_step: str = "scaler"
    model_step: str = "model"

    def __init__(
        self,
        *,
        model_params: P,
        model_search_space: S,
        scaler_params: ScalerParams = ScalerParams(),
    ):
        self.model_params = model_params
        self.model_search_space = model_search_space
        self.scaler_params = scaler_params

    @property
    def params(self) -> P:  # unprefixed model params
        return self.model_params

    @property
    def search_space(self) -> S:  # unprefixed model search space
        return self.model_search_space

    def _make_scaler(self) -> TransformerMixin:
        sp = self.scaler_params
        t = sp.scaler_type
        if t is ScalerType.NONE:
            return NoOpTransformer()
        if t is ScalerType.STANDARD:
            return StandardScaler(with_mean=sp.with_mean, with_std=sp.with_std)
        if t is ScalerType.MINMAX:
            return MinMaxScaler(feature_range=sp.feature_range)
        else:
            raise ValueError(f"Unsupported ScalerType: {t}")

    @abstractmethod
    def _make_model(
        self, model_params: dict[str, Any], *, random_state: int | None
    ) -> BaseEstimator: ...

    def build_estimator(
        self,
        params: dict[str, Any] | None = None,  # UNPREFIXED model param overrides
        *,
        random_state: int | None = None,
    ) -> Pipeline:
        mp = dc_to_dict(self.model_params)
        if params:
            mp.update(params)
        return Pipeline(
            [
                (self.scaler_step, self._make_scaler()),
                (self.model_step, self._make_model(mp, random_state=random_state)),
            ]
        )

    def get_prefixed_search_space(self) -> dict[str, Any]:
        # Only model search space is tunable
        return prefix(self.model_step, dc_to_dict(self.model_search_space))

    # ---- CV helpers (model-focused) ----
    def normalize_model_params_for_aggregation(
        self, best_params_prefixed: dict[str, Any]
    ) -> dict[str, Any]:
        return unprefix(self.model_step, best_params_prefixed)

    def _aggregate_params(self, params_list):
        if not params_list:
            return {}
        bucket = {}
        for d in params_list:
            for k, v in d.items():
                bucket.setdefault(k, []).append(v)
        n = len(params_list)
        out = {}
        for k, vals in bucket.items():
            if len(vals) < (n + 1) // 2:
                continue
            if all(is_bool(v) for v in vals):
                out[k] = mode([bool(v) for v in vals])
            elif all(is_numeric_ex_bool(v) for v in vals):
                out[k] = float(np.median(vals))
            else:
                out[k] = mode(vals)
        return out

    def _finalize_params(self, aggregated_params):
        defaults = dc_to_dict(self.model_params)
        final = dict(defaults)
        for k, v in aggregated_params.items():
            if k not in defaults:
                final[k] = v
                continue
            dv = defaults[k]
            if isinstance(dv, (bool, np.bool_)):
                final[k] = bool(v)
            elif isinstance(dv, (np.integer, int)) and isinstance(
                v, (float, int, np.floating, np.integer)
            ):
                final[k] = int(round(float(v)))
            elif isinstance(dv, (np.floating, float)) and isinstance(
                v, (float, int, np.floating, np.integer)
            ):
                final[k] = float(v)
            elif dv is None and isinstance(v, (float, int, np.floating, np.integer)):
                f = float(v)
                final[k] = int(round(f)) if f.is_integer() else f
            else:
                final[k] = v
        return final


# ---------- Concrete learners ----------
# Ridge
@dataclass(frozen=True)
class RidgeParams:
    alpha: float = 1.0
    fit_intercept: bool = True


@dataclass(frozen=True)
class RidgeSearchSpace:
    alpha: Any = field(default_factory=lambda: loguniform(1e-6, 1e3))
    fit_intercept: list[bool] = field(default_factory=lambda: [True, False])


class RidgeLearner(Learner[RidgeParams, RidgeSearchSpace]):
    name = "ridge"

    def __init__(
        self,
        params: RidgeParams = RidgeParams(),
        search_space: RidgeSearchSpace = RidgeSearchSpace(),
        scaler_params: ScalerParams = ScalerParams(),
    ):
        super().__init__(
            model_params=params,
            model_search_space=search_space,
            scaler_params=scaler_params,
        )

    def _make_model(
        self, model_params: dict[str, Any], *, random_state: int | None
    ) -> BaseEstimator:
        return Ridge(**model_params)  # random_state unused


# Random Forest
@dataclass(frozen=True)
class RFParams:
    n_estimators: int = 100
    max_depth: int | None = None
    max_features: Any = "sqrt"
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    bootstrap: bool = True
    n_jobs: int = -1


@dataclass(frozen=True)
class RFSearchSpace:
    n_estimators: Any = field(default_factory=lambda: randint(300, 1001))
    max_depth: list[int | None] = field(
        default_factory=lambda: [None, 6, 10, 16, 24, 32]
    )
    max_features: list[Any] = field(
        default_factory=lambda: ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]
    )
    min_samples_split: Any = field(default_factory=lambda: randint(2, 21))
    min_samples_leaf: Any = field(default_factory=lambda: randint(5, 30))
    bootstrap: list[bool] = field(default_factory=lambda: [True, False])


class RandomForestLearner(Learner[RFParams, RFSearchSpace]):
    name = "random_forest"

    def __init__(
        self,
        params: RFParams = RFParams(),
        search_space: RFSearchSpace = RFSearchSpace(),
        scaler_params: ScalerParams = ScalerParams(),
    ):
        super().__init__(
            model_params=params,
            model_search_space=search_space,
            scaler_params=scaler_params,
        )

    def _make_model(
        self, model_params: dict[str, Any], *, random_state: int | None
    ) -> BaseEstimator:
        if random_state is not None:
            model_params = dict(model_params, random_state=int(random_state))
        return RandomForestRegressor(**model_params)


# LightGBM
@dataclass(frozen=True)
class LGBMParams:
    n_estimators: int = 4000
    learning_rate: float = 0.03
    num_leaves: int = 256
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    boosting_type: str = "gbdt"
    n_jobs: int = -1
    verbosity: int = -1
    early_stopping_round = 100


@dataclass(frozen=True)
class LGBMSearchSpace:
    n_estimators: Any = field(default_factory=lambda: randint(400, 1001))
    learning_rate: Any = field(default_factory=lambda: loguniform(1e-3, 2e-1))
    num_leaves: Any = field(default_factory=lambda: randint(31, 63))
    max_depth: list[int] = field(default_factory=lambda: [-1, 2,4,6, 8])
    min_child_samples: Any = field(default_factory=lambda: randint(5, 201))
    subsample: Any = field(default_factory=lambda: uniform(0.5, 0.5))
    colsample_bytree: Any = field(default_factory=lambda: uniform(0.5, 0.5))
    reg_alpha: Any = field(default_factory=lambda: loguniform(1e-3, 10.0))
    reg_lambda: Any = field(default_factory=lambda: loguniform(1e-3, 10.0))
    boosting_type: list[str] = field(default_factory=lambda: ["gbdt", "goss"])


class LightGBMLearner(Learner[LGBMParams, LGBMSearchSpace]):
    name = "lightgbm"

    def __init__(
        self,
        params: LGBMParams = LGBMParams(),
        search_space: LGBMSearchSpace = LGBMSearchSpace(),
        scaler_params: ScalerParams = ScalerParams(),
    ):
        super().__init__(
            model_params=params,
            model_search_space=search_space,
            scaler_params=scaler_params,
        )

    def _make_model(
        self, model_params: dict[str, Any], *, random_state: int | None
    ) -> BaseEstimator:
        if random_state is not None:
            model_params = dict(model_params, random_state=int(random_state))
        return LGBMRegressor(**model_params)
