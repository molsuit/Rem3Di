from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer


class LatentCapacityReportModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    H_tot: float
    utilisation: float
    dead_dims: int
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
                "drop_outlier_quantile must be in (0, 1]; "
                f"got {drop_outlier_quantile}"
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

    N, d = Z.shape

    # normalize bins parameter -> per-dimension integer array
    if isinstance(bins, Iterable) and not isinstance(bins, str | bytes):
        bins_per_dim = np.array(list(bins), dtype=int)
        if bins_per_dim.shape[0] != d:
            raise ValueError("len(bins) must equal latent dimension d")
    else:
        bins_per_dim = np.full(d, int(bins), dtype=int)

    bins_per_dim = np.clip(bins_per_dim, 2, None)
    max_bits = np.log2(bins_per_dim)

    # per-dimension marginal entropies
    H_i = np.empty(d, dtype=np.float64)
    for j in range(d):
        counts, _ = np.histogram(Z[:, j], bins=bins_per_dim[j])
        total = counts.sum()
        if total == 0:
            H_i[j] = 0.0
            continue
        p = counts / (total + eps)
        H_i[j] = -np.sum(np.where(p > 0, p * np.log2(p), 0.0))

    H_tot = float(H_i.sum())

    # covariance spectrum and effective dimension (participation ratio)
    cov = np.cov(Z, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    num = (eigvals.sum()) ** 2
    den = np.square(eigvals).sum() + eps
    d_eff = float(num / den)

    # capacity proxy and utilisation
    C_eff = float(d_eff * np.mean(max_bits))
    utilisation = float(H_tot / (C_eff + eps))

    # dead dimensions
    dead_dims = int(np.sum(H_i < dead_thr * max_bits))

    explained_var_ratio = eigvals / (eigvals.sum() + eps)

    return LatentCapacityReportModel(
        H_tot=H_tot,
        utilisation=utilisation,
        dead_dims=dead_dims,
        n_samples_used=int(Z.shape[0]),
        n_dropped=int(n_input - Z.shape[0]),
        norm_threshold=norm_threshold,
        H_per_dim=H_i,
        H_norm_per_dim=H_i / (max_bits + eps),
        eigvals=eigvals,
        d_eff=d_eff,
        max_bits_per_dim=max_bits,
        explained_var_ratio=explained_var_ratio,
    )


def get_descriptor_norm_distribution(Z):
    Z = np.asarray(Z, dtype=np.float64)
    norms = np.linalg.norm(Z, axis=-1)
    print(norms.shape)
    return norms


def get_descriptor_channel_distribution(Z):
    Z = np.asarray(Z, dtype=np.float64)
    N, d = Z.shape

    means = np.mean(Z, axis=0)
    stds = np.std(Z, axis=0)
    print(means.shape)

    return means, stds
