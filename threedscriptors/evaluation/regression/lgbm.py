from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb  # for callbacks. If you prefer, you can omit callbacks entirely.
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
    train_test_split,
)
from tqdm import tqdm
from threedscriptors.evaluation.regression.learner import Learner

from threedscriptors.evaluation.regression.search_space import LogUniform, IntRange, Choice, FloatRange

from threedscriptors.evaluation.regression.scaling import AutoScaler

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

    def to_dict(self) -> dict[str, Any]:
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
        }
        # Drop None values
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class LGBMSearchSpace:
    learning_rate =  LogUniform(1e-3, 3e-1),
    num_leaves =  IntRange(16, 512, inclusive=True),
    max_depth =  Choice([-1, 6, 8, 10, 12, 14, 16]),
    min_child_samples =  IntRange(5, 100, inclusive=True),
    subsample =  FloatRange(0.5, 1.0),
    colsample_bytree = FloatRange(0.5, 1.0),
    reg_lambda =  LogUniform(1e-4, 30.0),
    reg_alpha =  Choice([0.0, 1e-3, 1e-2, 1e-1, 0.5, 1.0]),
    n_estimators =  IntRange(800, 6000, inclusive=True),
    boosting_type =  Choice(["gbdt"]),  # keep simple; add "goss" if you handle its constraints

    

class LGBMLearner(Learner):

    def __init__(self, params : LGBMParams | None = None, search_space: LGBMSearchSpace | None = None):

        self.params = params if params is not None else LGBMParams()
        self.search_space = search_space if search_space is not None else LGBMSearchSpace

    def name(self) -> str: return "lgbm"


    def build_estimator(self, params: dict[str, Any] | None = None, *, random_state: int | None = None):
        p = LGBMParams(**{**LGBMParams().to_dict(), **(params or {})})
        p = p.merge({})  # validate/cast
        from lightgbm import LGBMRegressor
        # keep a unified ("pre", "model") pipeline
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("pre", AutoScaler(strategy="none")),
            ("model", LGBMRegressor(**{**p.to_dict(), "random_state": random_state}))  # random_state override
        ])