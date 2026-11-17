"""BitBIRCH clustering of packed fingerprints and the UMAP of a sample of them.

One implementation for the single-dataset analysis (``analysis.py``) and the
dataset comparison (``comparison.py``): the threshold (configured or derived
from the fingerprints), the fit, the row-to-cluster assignment, the cluster-size
histogram, the summary, and the UMAP of a sample of the rows chosen by the
configured :data:`~remedi_eval.dataset_analysis.config.UmapSampling`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field

from remedi_eval.dataset_analysis.config import BitBirchConfig, BitBirchUmapConfig
from remedi_eval.dataset_analysis.statistics import DistributionStats
from remedi_eval.projection import UmapConfig

log = logging.getLogger(__name__)


class BitBirchUmapSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    n_points: int = 0
    projection: UmapConfig | None = None
    top_clusters_colored: int = 0
    fit_seconds: float | None = None
    notes: str | None = None


class BitBirchSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    n_input_smiles: int = 0
    n_fingerprints: int = 0
    n_clusters: int = 0
    singleton_clusters: int = 0
    largest_cluster: int = 0
    cluster_size_distribution: DistributionStats = Field(
        default_factory=lambda: DistributionStats(n=0)
    )
    threshold_used: float | None = None
    """Actual threshold passed to BitBIRCH (resolved from auto-calibration
    when ``BitBirchConfig.auto_threshold`` is true, else the config value)."""
    threshold_source: Literal["config", "auto"] | None = None
    mean_isim: float | None = None
    """Mean pairwise Tanimoto (iSIM) of the fingerprint set, when computed."""
    std_isim: float | None = None
    """Estimated std of pairwise Tanimoto, when computed."""
    umap: BitBirchUmapSummary | None = None
    notes: str | None = None


def resolve_bitbirch_threshold(
    config: BitBirchConfig, fingerprints_packed: np.ndarray
) -> tuple[float, Literal["config", "auto"], float | None, float | None]:
    """Decide the BitBIRCH threshold for a packed-byte fingerprint matrix.

    Returns ``(threshold, source, mean_isim, std_isim)``. When
    ``config.auto_threshold`` is false the source is ``"config"`` and the
    iSIM/std fields are ``None``; otherwise ``bblean.bitbirch.guess_threshold``
    (mean iSIM + factor·std) decides, and the numbers are returned so callers
    can record what was used."""
    if not config.auto_threshold:
        return float(config.threshold), "config", None, None
    import bblean.bitbirch as bb

    threshold, mean, std = bb.guess_threshold(
        fingerprints_packed,
        input_is_packed=True,
        n_features=config.fingerprint_size,
        factor=config.auto_threshold_factor,
        return_mean_std=True,
    )
    threshold = float(np.clip(threshold, 0.0, 1.0))
    return threshold, "auto", float(mean), float(std)


@dataclass(frozen=True)
class BitBirchClusters:
    """A BitBIRCH fit: the cluster of every fingerprint row and the cluster sizes."""

    row_cluster: np.ndarray
    sizes: np.ndarray
    threshold: float
    threshold_source: Literal["config", "auto"]
    mean_isim: float | None
    std_isim: float | None

    def largest(self, count: int) -> np.ndarray:
        """The ids of the ``count`` largest clusters, largest first."""
        count = min(count, int(self.sizes.size))
        if count <= 0:
            return np.asarray([], dtype=np.int64)
        return np.argsort(self.sizes)[-count:][::-1]

    def size_histogram(self, title: str) -> Figure:
        figure, axes = plt.subplots(figsize=(8, 4))
        if self.sizes.size:
            axes.hist(
                self.sizes,
                bins=min(50, max(5, int(np.sqrt(self.sizes.size)))),
                color="#8172b3",
            )
            axes.set_yscale("log")
        axes.set_xlabel("Molecules per cluster")
        axes.set_ylabel("Cluster count (log)")
        axes.set_title(
            f"{title} (n={self.sizes.size}, threshold={self.threshold:.2f}"
            f" [{self.threshold_source}])"
        )
        figure.tight_layout()
        return figure

    def summary(
        self, n_input_smiles: int, umap: BitBirchUmapSummary | None
    ) -> BitBirchSummary:
        return BitBirchSummary(
            enabled=True,
            n_input_smiles=n_input_smiles,
            n_fingerprints=int(self.row_cluster.size),
            n_clusters=int(self.sizes.size),
            singleton_clusters=int((self.sizes == 1).sum()),
            largest_cluster=int(self.sizes.max()) if self.sizes.size else 0,
            cluster_size_distribution=DistributionStats.from_array(self.sizes),
            threshold_used=self.threshold,
            threshold_source=self.threshold_source,
            mean_isim=self.mean_isim,
            std_isim=self.std_isim,
            umap=umap,
        )


def fit_bitbirch(
    fingerprints_packed: np.ndarray, config: BitBirchConfig
) -> BitBirchClusters:
    """Cluster the rows of a packed-byte fingerprint matrix."""
    import bblean

    threshold, source, mean_isim, std_isim = resolve_bitbirch_threshold(
        config, fingerprints_packed
    )
    log.info(
        "BitBIRCH: %d fingerprints, threshold=%.3f (source=%s, mean_iSIM=%s, std=%s)",
        len(fingerprints_packed),
        threshold,
        source,
        f"{mean_isim:.3f}" if mean_isim is not None else "n/a",
        f"{std_isim:.3f}" if std_isim is not None else "n/a",
    )
    tree = bblean.BitBirch(
        threshold=threshold,
        branching_factor=config.branching_factor,
        merge_criterion=config.merge_criterion,
        tolerance=config.tolerance,
    )
    tree.fit(fingerprints_packed)
    clusters = tree.get_cluster_mol_ids()
    row_cluster = np.full(len(fingerprints_packed), -1, dtype=np.int64)
    for cluster_id, rows in enumerate(clusters):
        row_cluster[np.asarray(rows, dtype=np.int64)] = cluster_id
    return BitBirchClusters(
        row_cluster=row_cluster,
        sizes=np.asarray([len(rows) for rows in clusters], dtype=np.int64),
        threshold=threshold,
        threshold_source=source,
        mean_isim=mean_isim,
        std_isim=std_isim,
    )


def umap_sample_rows(
    clusters: BitBirchClusters, config: BitBirchUmapConfig, rng: np.random.Generator
) -> np.ndarray:
    """The sorted rows the UMAP is fitted on, at most ``config.sample_size``.

    ``uniform`` draws them uniformly. ``largest_clusters_first`` keeps members
    of the ``top_clusters_colored`` largest clusters (up to half the budget) and
    fills the rest uniformly, so the coloured clusters stay visible when the
    budget is far below the number of fingerprints.
    """
    n_rows = int(clusters.row_cluster.size)
    budget = n_rows if config.sample_size is None else min(config.sample_size, n_rows)
    if config.sampling == "uniform":
        if budget >= n_rows:
            return np.arange(n_rows, dtype=np.int64)
        return np.sort(rng.choice(n_rows, size=budget, replace=False))
    top = clusters.largest(config.top_clusters_colored)
    top_rows = np.flatnonzero(np.isin(clusters.row_cluster, top))
    top_budget = min(top_rows.size, max(1, budget // 2))
    if top_rows.size > top_budget:
        top_rows = rng.choice(top_rows, size=top_budget, replace=False)
    rest_rows = np.setdiff1d(np.arange(n_rows), top_rows)
    rest_budget = max(0, budget - int(top_rows.size))
    if rest_rows.size > rest_budget:
        rest_rows = rng.choice(rest_rows, size=rest_budget, replace=False)
    return np.sort(np.concatenate([top_rows, rest_rows]).astype(np.int64))


@dataclass(frozen=True)
class UmapFit:
    """The 2D coordinates of the sampled rows and the largest clusters' ids."""

    rows: np.ndarray
    coordinates: np.ndarray
    top_cluster_ids: np.ndarray


def fit_umap(
    fingerprints_packed: np.ndarray,
    clusters: BitBirchClusters,
    config: BitBirchUmapConfig,
    fingerprint_size: int,
    rng: np.random.Generator,
) -> tuple[UmapFit | None, BitBirchUmapSummary]:
    """UMAP of a sample of the fingerprints (unpacked bits).

    Returns the fit (``None`` when disabled, empty or failed) and its summary;
    a failed fit is reported in the summary's ``notes``, not raised.
    """
    if not config.enabled:
        return None, BitBirchUmapSummary(enabled=False, notes="disabled by config")
    if len(fingerprints_packed) == 0:
        return None, BitBirchUmapSummary(enabled=True, notes="no fingerprints")
    rows = umap_sample_rows(clusters, config, rng)
    bits = np.unpackbits(fingerprints_packed[rows], axis=1, count=fingerprint_size)
    log.info(
        "BitBIRCH UMAP: fitting %d points (%s, sampling=%s)",
        rows.size,
        config.projection.label(),
        config.sampling,
    )
    started = time.perf_counter()
    try:
        coordinates = config.projection.project(bits.astype(np.float32))
    except Exception as exc:
        return None, BitBirchUmapSummary(
            enabled=True, n_points=int(rows.size), notes=f"UMAP fit failed: {exc!r}"
        )
    top = clusters.largest(config.top_clusters_colored)
    return UmapFit(
        rows=rows, coordinates=coordinates, top_cluster_ids=top
    ), BitBirchUmapSummary(
        enabled=True,
        n_points=int(rows.size),
        projection=config.projection,
        top_clusters_colored=int(top.size),
        fit_seconds=time.perf_counter() - started,
    )
