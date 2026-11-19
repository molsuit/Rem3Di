"""A fixed linear projection of pooled descriptors: per-channel standardisation
followed by principal components, fitted once on a reference set (e.g. the
pretraining corpus) and stored as an ``.npz`` file.

It makes the mean-pooled MACE baseline comparable in dimension to a learned
descriptor (:class:`~remedi.embedding.calculator.MaceMeanProjectionConfig`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Channels whose deviation is below this are only centred.
MINIMUM_STANDARD_DEVIATION = 1e-8


@dataclass(frozen=True)
class PooledProjection:
    mean: np.ndarray
    """``(D,)`` per-channel mean of the reference set."""
    standard_deviation: np.ndarray
    """``(D,)`` per-channel deviation; 1 for a constant channel."""
    components: np.ndarray
    """``(K, D)`` principal axes of the standardised reference set, by
    decreasing variance."""
    explained_variance_ratio: np.ndarray
    """``(K,)`` share of the standardised variance on each axis."""

    @property
    def dimension(self) -> int:
        return int(self.components.shape[0])

    def apply(self, values: np.ndarray) -> np.ndarray:
        """``(N, D)`` -> ``(N, K)`` float32."""
        standardised = (values - self.mean) / self.standard_deviation
        return (standardised @ self.components.T).astype(np.float32)

    def save(self, path: Path) -> None:
        np.savez(
            path,
            mean=self.mean,
            standard_deviation=self.standard_deviation,
            components=self.components,
            explained_variance_ratio=self.explained_variance_ratio,
        )

    @classmethod
    def load(cls, path: Path) -> PooledProjection:
        """Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the arrays do not fit together.
        """
        with np.load(path) as arrays:
            projection = cls(
                mean=arrays["mean"],
                standard_deviation=arrays["standard_deviation"],
                components=arrays["components"],
                explained_variance_ratio=arrays["explained_variance_ratio"],
            )
        width = projection.mean.shape[0]
        if projection.standard_deviation.shape != (width,) or (
            projection.components.ndim != 2 or projection.components.shape[1] != width
        ):
            raise ValueError(f"{path} holds arrays of mismatched shapes")
        return projection


def fit_pooled_projection(values: np.ndarray, dimension: int) -> PooledProjection:
    """Standardise ``values`` ``(N, D)`` per channel and keep the top
    ``dimension`` principal components.

    Raises:
        ValueError: if ``dimension`` exceeds ``min(N, D)`` or values are not finite.
    """
    if values.ndim != 2:
        raise ValueError(f"expected (N, D) values, got shape {values.shape}")
    count, width = values.shape
    if not 0 < dimension <= min(count, width):
        raise ValueError(
            f"cannot keep {dimension} components of {count} samples x {width} channels"
        )
    if not np.isfinite(values).all():
        raise ValueError("the values are not finite")
    values = values.astype(np.float64)
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0)
    standard_deviation = np.where(
        standard_deviation < MINIMUM_STANDARD_DEVIATION, 1.0, standard_deviation
    )
    standardised = (values - mean) / standard_deviation
    _, singular_values, right_vectors = np.linalg.svd(standardised, full_matrices=False)
    variances = singular_values**2
    return PooledProjection(
        mean=mean,
        standard_deviation=standard_deviation,
        components=right_vectors[:dimension],
        explained_variance_ratio=variances[:dimension] / variances.sum(),
    )
