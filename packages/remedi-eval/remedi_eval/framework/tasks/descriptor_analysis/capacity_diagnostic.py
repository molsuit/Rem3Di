"""Capacity statistics of a descriptor matrix and their plots.

The per-dimension statistics (histogram entropy, covariance spectrum,
participation ratio, explained variance, dead dimensions) are shared with the
streaming MACE invariant analysis in :mod:`mace_invariant_stats`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer
from scipy.stats import entropy


def marginal_entropy_bits(
    samples: np.ndarray,
    bins: np.ndarray,
    value_ranges: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Histogram entropy in bits of every column of ``samples`` (rows are samples).

    ``bins`` holds the bin count per column. ``value_ranges`` (lower, upper per
    column) fixes the histogram range; by default each column spans its own
    minimum to maximum. A column with an empty range or no samples has entropy 0.
    """
    dimension = samples.shape[1]
    entropies = np.zeros(dimension, dtype=np.float64)
    for column in range(dimension):
        histogram_range = None
        if value_ranges is not None:
            lower = float(value_ranges[0][column])
            upper = float(value_ranges[1][column])
            if upper <= lower:
                continue
            histogram_range = (lower, upper)
        counts, _ = np.histogram(
            samples[:, column], bins=int(bins[column]), range=histogram_range
        )
        if counts.sum() > 0:
            entropies[column] = float(entropy(counts, base=2))
    return entropies


def covariance_spectrum(covariance: np.ndarray) -> np.ndarray:
    """Eigenvalues of a covariance matrix, descending, negative round-off clipped to 0."""
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    return np.clip(eigenvalues, 0.0, None)


def participation_ratio(eigenvalues: np.ndarray, eps: float = 1e-12) -> float:
    """Effective dimension ``(sum eigenvalues)^2 / sum eigenvalues^2``."""
    total = eigenvalues.sum()
    return float((total * total) / (np.square(eigenvalues).sum() + eps))


def explained_variance_ratio(eigenvalues: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return eigenvalues / (eigenvalues.sum() + eps)


def count_dead_dimensions(
    entropy_bits: np.ndarray, max_bits: np.ndarray | float, threshold: float
) -> int:
    """Dimensions whose entropy is below ``threshold`` of their maximum."""
    return int(np.sum(entropy_bits < threshold * np.asarray(max_bits)))


def draw_per_dimension_entropy(
    axis: Any,
    normalized_entropy: np.ndarray,
    dead_threshold: float,
    color: Any = None,
) -> None:
    """Bar chart of the normalised per-dimension entropy, sorted, with the dead line."""
    order = np.argsort(normalized_entropy)[::-1]
    axis.bar(np.arange(len(order)), normalized_entropy[order], color=color, width=1.0)
    axis.axhline(
        dead_threshold,
        color="red",
        linestyle="--",
        linewidth=1,
        label="dead threshold",
    )
    axis.set_xlabel("Dimension (sorted)")
    axis.legend(fontsize="small")


def draw_cumulative_explained_variance(
    axis: Any,
    explained_variance: np.ndarray,
    label: str | None = None,
    color: Any = None,
) -> None:
    cumulative = np.cumsum(explained_variance)
    axis.plot(
        np.arange(1, len(cumulative) + 1),
        cumulative,
        marker=".",
        color=color,
        label=label,
    )
    axis.set_xlabel("PCA component")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.3)


class LatentCapacityReportModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    H_tot: float
    utilisation: float
    dead_dims: int
    dead_threshold: float
    d_eff: float

    n_samples_used: int
    n_dropped: int = 0
    norm_threshold: float | None = None

    H_per_dim: np.ndarray = Field(repr=False)
    H_norm_per_dim: np.ndarray = Field(repr=False)
    eigvals: np.ndarray = Field(repr=False)
    max_bits_per_dim: np.ndarray = Field(repr=False)
    explained_var_ratio: np.ndarray = Field(repr=False)

    # Ensure np.ndarray → list for YAML (via model_dump(mode="json"))
    @field_serializer(
        "H_per_dim",
        "H_norm_per_dim",
        "eigvals",
        "max_bits_per_dim",
        "explained_var_ratio",
        when_used="json",
    )
    def _serialize_ndarray(self, v: np.ndarray) -> list[float]:
        return v.tolist()


def run_latent_space_capacity_diagnostic(
    Z: np.ndarray,
    bins: int | Iterable[int] = 128,
    dead_thr: float = 0.2,
    eps: float = 1e-12,
    *,
    standardize: bool = False,
    l2_normalize: bool = False,
    drop_outlier_quantile: float | None = None,
) -> LatentCapacityReportModel:
    """
    Fixed-bin marginal entropy diagnostic with covariance-based effective dimension.
    No bias correction; no alternative binning rules.

    Parameters
    ----------
    Z : array_like, shape (N, d)
        Latent vectors (rows are samples).
    bins : int or iterable[int]
        Number of histogram bins per dimension (>=2). If iterable, len == d.
    dead_thr : float
        Flag dim as 'dead' if H_i < dead_thr * log2(bins_i).
    eps : float
        Numerical jitter for stability.
    standardize : bool
        If True, z-score each dimension before analysis.
    l2_normalize : bool
        If True, project each row onto the unit sphere before analysis. Applied
        after standardization (when both are set), so the diagnostic measures
        the capacity of the *direction* of the descriptor — useful when norm
        outliers dominate the covariance spectrum. Combining with
        ``standardize`` is uncommon; the typical use is one or the other.
    drop_outlier_quantile : float in (0, 1] or None
        If set, drop samples whose raw L2 norm exceeds this quantile before any
        centering / normalization, e.g. ``0.99`` removes the top 1% of high-norm
        outliers. Useful when a handful of pathological samples (under-coordinated
        structures, missing-H artifacts, etc.) dominate the covariance spectrum.

    Returns
    -------
    LatentCapacityReport
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("Z must be 2D (N, d)")

    n_input = Z.shape[0]
    norm_threshold: float | None = None
    if drop_outlier_quantile is not None:
        if not 0.0 < drop_outlier_quantile <= 1.0:
            raise ValueError(
                f"drop_outlier_quantile must be in (0, 1]; got {drop_outlier_quantile}"
            )
        raw_norms = np.linalg.norm(Z, axis=1)
        norm_threshold = float(np.quantile(raw_norms, drop_outlier_quantile))
        Z = Z[raw_norms <= norm_threshold]
        if Z.shape[0] < 2:
            raise ValueError(
                "Outlier filter removed too many samples; "
                f"only {Z.shape[0]} remain after dropping > q={drop_outlier_quantile}"
            )

    # center; (optional) standardize; (optional) L2-normalize per row
    Z = Z - Z.mean(axis=0, keepdims=True)
    if standardize:
        s = Z.std(axis=0, ddof=1)
        s[s < eps] = 1.0
        Z = Z / s
    if l2_normalize:
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        Z = Z / (norms + eps)

    _n_structures, d = Z.shape

    # normalize bins parameter -> per-dimension integer array
    if isinstance(bins, Iterable) and not isinstance(bins, str | bytes):
        bins_per_dim = np.array(list(bins), dtype=int)
        if bins_per_dim.shape[0] != d:
            raise ValueError("len(bins) must equal latent dimension d")
    else:
        bins_per_dim = np.full(d, int(bins), dtype=int)

    bins_per_dim = np.clip(bins_per_dim, 2, None)
    max_bits = np.log2(bins_per_dim)

    H_i = marginal_entropy_bits(Z, bins_per_dim)
    H_tot = float(H_i.sum())

    # covariance spectrum and effective dimension (participation ratio)
    eigvals = covariance_spectrum(np.cov(Z, rowvar=False))
    d_eff = participation_ratio(eigvals, eps)

    # capacity proxy and utilisation
    C_eff = float(d_eff * np.mean(max_bits))
    utilisation = float(H_tot / (C_eff + eps))

    return LatentCapacityReportModel(
        H_tot=H_tot,
        utilisation=utilisation,
        dead_dims=count_dead_dimensions(H_i, max_bits, dead_thr),
        dead_threshold=dead_thr,
        n_samples_used=int(Z.shape[0]),
        n_dropped=int(n_input - Z.shape[0]),
        norm_threshold=norm_threshold,
        H_per_dim=H_i,
        H_norm_per_dim=H_i / (max_bits + eps),
        eigvals=eigvals,
        d_eff=d_eff,
        max_bits_per_dim=max_bits,
        explained_var_ratio=explained_variance_ratio(eigvals, eps),
    )


def get_descriptor_norm_distribution(Z):
    Z = np.asarray(Z, dtype=np.float64)
    return np.linalg.norm(Z, axis=-1)


def get_descriptor_channel_distribution(Z):
    Z = np.asarray(Z, dtype=np.float64)
    _n_structures, _dimension = Z.shape  # Z must be 2-D
    return np.mean(Z, axis=0), np.std(Z, axis=0)
