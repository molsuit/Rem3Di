from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from sklearn.base import BaseEstimator



class Learner(ABC):
    """Abstract base for regression learners."""
    params: Dict
    search_space: Dict
    name: str


    @abstractmethod
    def build_estimator(self, params: Optional[Dict[str, Any]] = None,
                        *, random_state: Optional[int] = None) -> BaseEstimator:
        """
        Return a fresh, unfitted estimator (may be a Pipeline). `params` are unprefixed.
        Implementations must internally apply any pipeline prefixes and ignore
        unsupported keys (e.g. random_state for Ridge).
        """
        ...

    @staticmethod
    def normalize_model_params_for_aggregation(params: dict[str, Any]) -> dict[str, Any]:
        # deprefix and keep only model params
        out = {}
        for k, v in params.items():
            if "__" in k:
                step, key = k.split("__", 1)
                if step != "model":
                    continue
                out[key] = v
            else:
                out[k] = v
        return out