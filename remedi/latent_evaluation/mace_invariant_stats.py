"""Streaming statistics + information-theoretic metrics for atomwise MACE
invariant descriptors, with a layer-aware breakdown.

The analysis is intentionally generic: the layer grouping is recovered from the
e3nn ``Irreps`` of the MACE product stack (one invariant block per MACE layer),
so adding a third MACE layer would not require any change here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from e3nn.o3 import Irreps
from pydantic import BaseModel, ConfigDict, Field, field_serializer


class LayerSpec(BaseModel):
    """Mapping from MACE layer index to invariant index range."""

    layer_index: int
    start: int
    stop: int

    @property
    def width(self) -> int:
        return self.stop - self.start


def get_layer_invariant_specs(input_irreps: Irreps) -> list[LayerSpec]:
    """Return per-layer invariant index ranges within the *invariant-only*
    feature vector.

    Convention: the MACE product stack emits one ``Irreps`` block per layer
    that contains both invariant and equivariant parts. We treat each l=0
    block as a layer's invariant feature group. The returned slices are
    coordinates in the concatenated *invariant* tensor (after dropping the
    equivariant blocks), so they index directly into the per-atom invariant
    tensor produced by ``split_invariants_equivariants``.
    """
    specs: list[LayerSpec] = []
    cursor = 0
    layer_idx = 0
    for mul_ir in input_irreps:
        if mul_ir.ir.l == 0:
            width = mul_ir.dim
            specs.append(
                LayerSpec(layer_index=layer_idx, start=cursor, stop=cursor + width)
            )
            cursor += width
            layer_idx += 1
    if not specs:
        raise ValueError(
            f"No invariant (l=0) irreps found in input_irreps={input_irreps}"
        )
    return specs


class _Welford:
    """Per-dimension running mean / variance / min / max (numerically stable)."""

    def __init__(
        self, dim: int, device: torch.device, dtype: torch.dtype = torch.float64
    ):
        self.count = torch.zeros((), device=device, dtype=torch.float64)
        self.mean = torch.zeros(dim, device=device, dtype=dtype)
        self.M2 = torch.zeros(dim, device=device, dtype=dtype)
        self.min = torch.full((dim,), float("inf"), device=device, dtype=dtype)
        self.max = torch.full((dim,), float("-inf"), device=device, dtype=dtype)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        # x: (n, d) on same device
        n = x.shape[0]
        if n == 0:
            return
        x = x.to(self.mean.dtype)
        new_count = self.count + n
        batch_mean = x.mean(dim=0)
        delta = batch_mean - self.mean
        self.mean += delta * (n / new_count)
        # batch sum of squared deviations from batch mean
        batch_M2 = ((x - batch_mean) ** 2).sum(dim=0)
        self.M2 += batch_M2 + delta.pow(2) * (self.count.item() * n / new_count)
        self.count = new_count
        self.min = torch.minimum(self.min, x.amin(dim=0))
        self.max = torch.maximum(self.max, x.amax(dim=0))

    @property
    def variance(self) -> torch.Tensor:
        n = float(self.count.item())
        if n < 2:
            return torch.zeros_like(self.M2)
        return self.M2 / (n - 1)

    @property
    def std(self) -> torch.Tensor:
        return self.variance.clamp_min(0).sqrt()


class _CovarianceAccumulator:
    """Accumulates first and second moments to produce a sample covariance."""

    def __init__(
        self, dim: int, device: torch.device, dtype: torch.dtype = torch.float64
    ):
        self.dim = dim
        self.count = torch.zeros((), device=device, dtype=torch.float64)
        self.sum = torch.zeros(dim, device=device, dtype=dtype)
        self.outer = torch.zeros(dim, dim, device=device, dtype=dtype)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.shape[0] == 0:
            return
        x = x.to(self.sum.dtype)
        self.count += x.shape[0]
        self.sum += x.sum(dim=0)
        self.outer += x.t() @ x

    def covariance(self) -> torch.Tensor:
        n = float(self.count.item())
        if n < 2:
            return torch.zeros_like(self.outer)
        mean = self.sum / n
        cov = (self.outer - n * torch.outer(mean, mean)) / (n - 1)
        # Symmetrise to remove tiny numerical asymmetry
        return 0.5 * (cov + cov.t())


class _ReservoirSampler:
    """Vitter's Algorithm R for unbiased fixed-size sampling without replacement."""

    def __init__(self, capacity: int, dim: int, seed: int = 0):
        self.capacity = capacity
        self.dim = dim
        self.seen = 0
        self.buffer = np.empty((capacity, dim), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def update(self, x: np.ndarray) -> None:
        n = x.shape[0]
        if n == 0:
            return
        if self.seen < self.capacity:
            take = min(self.capacity - self.seen, n)
            self.buffer[self.seen : self.seen + take] = x[:take]
            self.seen += take
            x = x[take:]
            if x.shape[0] == 0:
                return
        # Replacement phase
        for row in x:
            self.seen += 1
            j = int(self.rng.integers(0, self.seen))
            if j < self.capacity:
                self.buffer[j] = row

    def sample(self) -> np.ndarray:
        n = min(self.seen, self.capacity)
        return self.buffer[:n]


class LayerStreamStats:
    """Streaming statistics for one MACE layer's invariants."""

    def __init__(
        self,
        spec: LayerSpec,
        device: torch.device,
        reservoir_capacity: int,
        seed: int,
    ):
        self.spec = spec
        self.welford = _Welford(spec.width, device=device)
        self.cov = _CovarianceAccumulator(spec.width, device=device)
        self.l2_norms: list[np.ndarray] = []
        # spec.layer_index can be -1 for the "total invariants" accumulator;
        # fold it into the seed via an unsigned hash so numpy's RNG accepts it.
        self.reservoir = _ReservoirSampler(
            capacity=reservoir_capacity,
            dim=spec.width,
            seed=abs(hash((int(seed), int(spec.layer_index)))) % (2**32),
        )

    def update(self, layer_invariants: torch.Tensor) -> None:
        # layer_invariants: (n_atoms, layer_width) on device
        self.welford.update(layer_invariants)
        self.cov.update(layer_invariants)
        layer_norms = torch.linalg.norm(layer_invariants.float(), dim=-1)
        self.l2_norms.append(layer_norms.detach().cpu().numpy())
        self.reservoir.update(layer_invariants.detach().cpu().float().numpy())


class LayerSummary(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    layer_index: int
    width: int
    n_atoms: int

    # Per-dim stats (length = width)
    mean_per_dim: np.ndarray = Field(repr=False)
    std_per_dim: np.ndarray = Field(repr=False)
    min_per_dim: np.ndarray = Field(repr=False)
    max_per_dim: np.ndarray = Field(repr=False)

    # L2 norm summary
    l2_norm_mean: float
    l2_norm_std: float
    l2_norm_min: float
    l2_norm_max: float

    # Information-theoretic / capacity metrics
    eigvals: np.ndarray = Field(repr=False)
    explained_var_ratio: np.ndarray = Field(repr=False)
    participation_ratio: float
    coding_rate: float
    coding_rate_eps: float
    mean_abs_correlation: float
    H_per_dim: np.ndarray = Field(repr=False)
    H_norm_per_dim: np.ndarray = Field(repr=False)
    H_total: float
    dead_dims: int

    @field_serializer(
        "mean_per_dim",
        "std_per_dim",
        "min_per_dim",
        "max_per_dim",
        "eigvals",
        "explained_var_ratio",
        "H_per_dim",
        "H_norm_per_dim",
        when_used="json",
    )
    def _serialize_ndarray(self, v: np.ndarray) -> list[float]:
        return v.tolist()


class TotalSummary(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_atoms: int
    invariant_dim: int

    l2_norm_mean: float
    l2_norm_std: float
    l2_norm_min: float
    l2_norm_max: float

    participation_ratio: float
    coding_rate: float
    coding_rate_eps: float
    mean_abs_correlation: float


class AnalysisReport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_irreps: str
    invariant_irreps: str
    layers: list[LayerSummary]
    total: TotalSummary
    histogram_bins: int


# ----------------------------------------------------------------------------
# Metric computations on a covariance matrix / reservoir sample
# ----------------------------------------------------------------------------


def participation_ratio(eigvals: np.ndarray, eps: float = 1e-12) -> float:
    s = eigvals.sum()
    return float((s * s) / (np.square(eigvals).sum() + eps))


def coding_rate(cov: np.ndarray, eps_sq: float, n_samples: int) -> float:
    """Lossy coding rate R(eps) = 0.5 log2 det(I + d/eps^2 * Cov).

    Substituting Z Z^T = m * Cov into Yu et al.'s
    R = 0.5 log det(I + (d/(m*eps^2)) Z Z^T) cancels the sample count m, so
    the formula written on the empirical covariance has no N inside it. The
    ``n_samples`` argument is retained only as documentation — it is not
    used for the rate itself, but a tiny minimum is enforced for safety.

    A higher coding rate means the covariance spectrum spans more "volume" in
    feature space — i.e. the descriptor is using more independent directions
    to a comparable magnitude.
    """
    if n_samples <= 0:
        return 0.0
    d = cov.shape[0]
    M = np.eye(d) + (d / eps_sq) * cov
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        # Numerical issue: fall back to the eigenvalue form which is always >=0
        eigvals = np.linalg.eigvalsh(M).clip(min=1e-30)
        logdet = float(np.log(eigvals).sum())
    return float(0.5 * logdet / np.log(2.0))


def mean_abs_correlation(cov: np.ndarray, eps: float = 1e-12) -> float:
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(std, std) + eps
    corr = cov / denom
    d = corr.shape[0]
    # Off-diagonal mean absolute correlation
    off_mask = ~np.eye(d, dtype=bool)
    return float(np.abs(corr[off_mask]).mean())


def per_dim_entropy(
    sample: np.ndarray,
    bins: int,
    range_min: np.ndarray,
    range_max: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram-based marginal entropy per dimension. Returns (H_per_dim,
    H_norm_per_dim) where the normalised version divides by log2(bins)."""
    _, d = sample.shape
    H = np.empty(d, dtype=np.float64)
    max_bits = float(np.log2(max(bins, 2)))
    for j in range(d):
        lo = float(range_min[j])
        hi = float(range_max[j])
        if hi - lo < eps:
            H[j] = 0.0
            continue
        counts, _ = np.histogram(sample[:, j], bins=bins, range=(lo, hi))
        total = counts.sum()
        if total == 0:
            H[j] = 0.0
            continue
        p = counts / (total + eps)
        H[j] = -np.sum(np.where(p > 0, p * np.log2(p), 0.0))
    H_norm = H / (max_bits + eps)
    return H, H_norm


# ----------------------------------------------------------------------------
# Top-level summarisation
# ----------------------------------------------------------------------------


def summarise_layer(
    stats: LayerStreamStats,
    *,
    histogram_bins: int,
    coding_rate_eps: float,
    dead_threshold: float,
) -> LayerSummary:
    n = int(stats.welford.count.item())
    mean = stats.welford.mean.detach().cpu().numpy().astype(np.float64)
    std = stats.welford.std.detach().cpu().numpy().astype(np.float64)
    mn = stats.welford.min.detach().cpu().numpy().astype(np.float64)
    mx = stats.welford.max.detach().cpu().numpy().astype(np.float64)

    norms = np.concatenate(stats.l2_norms) if stats.l2_norms else np.zeros(0)
    cov = stats.cov.covariance().detach().cpu().numpy().astype(np.float64)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    eigvals_clip = np.clip(eigvals, 0.0, None)
    pr = participation_ratio(eigvals_clip)
    expl = eigvals_clip / (eigvals_clip.sum() + 1e-12)

    sample = stats.reservoir.sample()
    H_per_dim, H_norm = per_dim_entropy(
        sample, bins=histogram_bins, range_min=mn, range_max=mx
    )
    H_tot = float(H_per_dim.sum())
    max_bits = float(np.log2(max(histogram_bins, 2)))
    dead = int(np.sum(H_per_dim < dead_threshold * max_bits))

    return LayerSummary(
        layer_index=stats.spec.layer_index,
        width=stats.spec.width,
        n_atoms=n,
        mean_per_dim=mean,
        std_per_dim=std,
        min_per_dim=mn,
        max_per_dim=mx,
        l2_norm_mean=float(norms.mean()) if norms.size else 0.0,
        l2_norm_std=float(norms.std()) if norms.size else 0.0,
        l2_norm_min=float(norms.min()) if norms.size else 0.0,
        l2_norm_max=float(norms.max()) if norms.size else 0.0,
        eigvals=eigvals,
        explained_var_ratio=expl,
        participation_ratio=pr,
        coding_rate=coding_rate(cov, coding_rate_eps**2, n),
        coding_rate_eps=coding_rate_eps,
        mean_abs_correlation=mean_abs_correlation(cov),
        H_per_dim=H_per_dim,
        H_norm_per_dim=H_norm,
        H_total=H_tot,
        dead_dims=dead,
    )


def summarise_total(
    total_stats: LayerStreamStats,
    *,
    coding_rate_eps: float,
) -> TotalSummary:
    n = int(total_stats.welford.count.item())
    norms = (
        np.concatenate(total_stats.l2_norms) if total_stats.l2_norms else np.zeros(0)
    )
    cov = total_stats.cov.covariance().detach().cpu().numpy().astype(np.float64)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    return TotalSummary(
        n_atoms=n,
        invariant_dim=total_stats.spec.width,
        l2_norm_mean=float(norms.mean()) if norms.size else 0.0,
        l2_norm_std=float(norms.std()) if norms.size else 0.0,
        l2_norm_min=float(norms.min()) if norms.size else 0.0,
        l2_norm_max=float(norms.max()) if norms.size else 0.0,
        participation_ratio=participation_ratio(eigvals),
        coding_rate=coding_rate(cov, coding_rate_eps**2, n),
        coding_rate_eps=coding_rate_eps,
        mean_abs_correlation=mean_abs_correlation(cov),
    )


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------


def _layer_color_cycle(n_layers: int) -> list[Any]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n_layers)]


def plot_mean_std_per_dim(report: AnalysisReport):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    colors = _layer_color_cycle(len(report.layers))

    cursor = 0
    for layer, color in zip(report.layers, colors, strict=True):
        x = np.arange(cursor, cursor + layer.width)
        axes[0].bar(
            x,
            layer.mean_per_dim,
            color=color,
            width=1.0,
            label=f"layer {layer.layer_index}",
        )
        axes[1].bar(
            x,
            layer.std_per_dim,
            color=color,
            width=1.0,
            label=f"layer {layer.layer_index}",
        )
        cursor += layer.width

    axes[0].set_ylabel("Mean (per dim)")
    axes[0].axhline(0.0, color="black", linewidth=0.6, linestyle="--")
    axes[0].set_title("Per-dimension mean of MACE invariant features (atomwise)")
    axes[0].legend(fontsize="small", ncol=len(report.layers))

    axes[1].set_ylabel("Std (per dim)")
    axes[1].set_xlabel("Invariant dimension index")
    axes[1].set_title("Per-dimension std of MACE invariant features (atomwise)")
    fig.tight_layout()
    return fig


def plot_l2_norm_per_layer(stats_list: list[LayerStreamStats], total_norms: np.ndarray):
    import matplotlib.pyplot as plt

    n_layers = len(stats_list)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = _layer_color_cycle(n_layers)

    # Histograms (overlaid, log-scale y)
    all_norms = [np.concatenate(s.l2_norms) for s in stats_list]
    lo = min(float(arr.min()) for arr in all_norms)
    hi = max(float(arr.max()) for arr in all_norms)
    bins = np.linspace(lo, hi, 80)
    for s, arr, color in zip(stats_list, all_norms, colors, strict=True):
        axes[0].hist(
            arr,
            bins=bins,
            histtype="step",
            color=color,
            label=f"layer {s.spec.layer_index} (μ={arr.mean():.2f})",
            linewidth=1.5,
        )
    axes[0].hist(
        total_norms,
        bins=np.linspace(float(total_norms.min()), float(total_norms.max()), 80),
        histtype="step",
        color="black",
        linestyle="--",
        label=f"total (μ={total_norms.mean():.2f})",
        linewidth=1.5,
    )
    axes[0].set_xlabel("Atomic invariant L2 norm")
    axes[0].set_ylabel("Atom count")
    axes[0].set_yscale("log")
    axes[0].set_title("Distribution of L2 norms")
    axes[0].legend(fontsize="small")

    # Violin per layer
    parts = axes[1].violinplot(
        all_norms,
        positions=[s.spec.layer_index for s in stats_list],
        showmeans=True,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_alpha(0.6)
    axes[1].set_xticks([s.spec.layer_index for s in stats_list])
    axes[1].set_xticklabels([f"layer {s.spec.layer_index}" for s in stats_list])
    axes[1].set_ylabel("Atomic invariant L2 norm")
    axes[1].set_title("Per-layer L2 norm distribution")

    fig.tight_layout()
    return fig


def plot_per_dim_entropy(report: AnalysisReport, dead_threshold: float):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(report.layers),
        figsize=(6 * len(report.layers), 4),
        sharey=True,
        squeeze=False,
    )
    colors = _layer_color_cycle(len(report.layers))
    for ax, layer, color in zip(axes[0], report.layers, colors, strict=True):
        order = np.argsort(layer.H_norm_per_dim)[::-1]
        ax.bar(
            np.arange(layer.width), layer.H_norm_per_dim[order], color=color, width=1.0
        )
        ax.axhline(
            dead_threshold,
            color="red",
            linestyle="--",
            linewidth=1,
            label="dead threshold",
        )
        ax.set_xlabel("Invariant dim (sorted)")
        ax.set_title(
            f"Layer {layer.layer_index}: H/log2(bins)\n"
            f"dead={layer.dead_dims}, "
            f"PR={layer.participation_ratio:.1f}, "
            f"R={layer.coding_rate:.1f} bits"
        )
        ax.legend(fontsize="small")
    axes[0][0].set_ylabel("Normalised marginal entropy")
    fig.tight_layout()
    return fig


def plot_explained_variance(report: AnalysisReport):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = _layer_color_cycle(len(report.layers))
    for layer, color in zip(report.layers, colors, strict=True):
        cum = np.cumsum(layer.explained_var_ratio)
        ax.plot(
            np.arange(1, len(cum) + 1),
            cum,
            color=color,
            label=f"layer {layer.layer_index} (PR={layer.participation_ratio:.1f})",
        )
    ax.set_xlabel("PCA component (per layer)")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-layer covariance spectrum of MACE invariants")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def write_report(report: AnalysisReport, output_dir: Path) -> None:
    """Persist plots + a YAML summary into ``output_dir``."""
    import pydantic_yaml as pyaml

    output_dir.mkdir(parents=True, exist_ok=True)
    pyaml.to_yaml_file(
        output_dir / "mace_invariant_report.yaml",
        report,
    )
