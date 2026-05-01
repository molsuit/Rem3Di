from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Annotated, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.visualize.plot import plot_atoms
from pydantic import BaseModel, ConfigDict, Field

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.evaluation.descriptor_analysis.capacity_diagnostic import (
    LatentCapacityReportModel,
    get_descriptor_norm_distribution,
    run_latent_space_capacity_diagnostic,
)
from threedscriptors.evaluation.descriptor_analysis.coloring import ColorProvider
from threedscriptors.evaluation.descriptor_analysis.context import (
    DescriptorAnalysisContext,
)
from threedscriptors.evaluation.results import (
    ChemiscopeResult,
    EvalResult,
    FigureResult,
    PydanticResult,
)


class _BaseAnalysisTask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def run(self, ctx: DescriptorAnalysisContext) -> list[EvalResult]: ...


class CapacityDiagnosticTask(_BaseAnalysisTask):
    """Marginal-entropy / participation-ratio diagnostic + summary figures."""

    kind: Literal["capacity_diagnostic"] = "capacity_diagnostic"
    bins: int = 128
    dead_threshold: float = 0.2
    standardize: bool = False
    l2_normalize: bool = False
    drop_outlier_quantile: float | None = None
    file_name: str = "capacity_diagnostic.yaml"
    plot_per_dim: bool = True
    plot_explained_variance: bool = True

    def run(self, ctx: DescriptorAnalysisContext) -> list[EvalResult]:
        report = run_latent_space_capacity_diagnostic(
            ctx.descriptors_raw,
            bins=self.bins,
            dead_thr=self.dead_threshold,
            standardize=self.standardize,
            l2_normalize=self.l2_normalize,
            drop_outlier_quantile=self.drop_outlier_quantile,
        )

        stem = Path(self.file_name).stem
        results: list[EvalResult] = [
            PydanticResult(file_name=ctx.file_name(self.file_name), obj=report),
        ]

        if self.plot_per_dim:
            results.append(
                FigureResult(
                    file_name=ctx.file_name(f"{stem}_per_dim.png"),
                    figure=_plot_per_dim_entropy(report),
                )
            )
        if self.plot_explained_variance:
            results.append(
                FigureResult(
                    file_name=ctx.file_name(f"{stem}_explained_variance.png"),
                    figure=_plot_explained_variance(report),
                )
            )

        return results


class DescriptorDistributionTask(_BaseAnalysisTask):
    """Element-wise value histogram and L2-norm histogram."""

    kind: Literal["descriptor_distribution"] = "descriptor_distribution"
    bins: int = 50
    log_scale: bool = True

    def run(self, ctx: DescriptorAnalysisContext) -> list[EvalResult]:
        results: list[EvalResult] = []

        flat = ctx.descriptors_raw.reshape(-1)
        fig_values, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            flat,
            bins=np.linspace(float(flat.min()), float(flat.max()), self.bins).tolist(),
        )
        if self.log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Descriptor value")
        ax.set_ylabel("Count")
        ax.set_title("Per-element descriptor value distribution")
        fig_values.tight_layout()
        results.append(
            FigureResult(
                file_name=ctx.file_name("descriptor_value_distribution.png"),
                figure=fig_values,
            )
        )

        norms = get_descriptor_norm_distribution(ctx.descriptors_raw)
        fig_norms, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            norms,
            bins=np.linspace(
                float(norms.min()), float(norms.max()), self.bins
            ).tolist(),
        )
        if self.log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("L2 norm")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of descriptor L2 norms")
        fig_norms.tight_layout()
        results.append(
            FigureResult(
                file_name=ctx.file_name("descriptor_norm_distribution.png"),
                figure=fig_norms,
            )
        )

        summary = _DescriptorDistributionSummary(
            n_samples=int(ctx.descriptors_raw.shape[0]),
            d_latent=int(ctx.descriptors_raw.shape[1]),
            value_mean=float(np.mean(ctx.descriptors_raw)),
            value_std=float(np.std(ctx.descriptors_raw)),
            value_min=float(np.min(ctx.descriptors_raw)),
            value_max=float(np.max(ctx.descriptors_raw)),
            norm_mean=float(np.mean(norms)),
            norm_std=float(np.std(norms)),
            norm_min=float(np.min(norms)),
            norm_max=float(np.max(norms)),
        )
        results.append(
            PydanticResult(
                file_name=ctx.file_name("descriptor_distribution_summary.yaml"),
                obj=summary,
            )
        )

        return results


class TopNormDescriptorsTask(_BaseAnalysisTask):
    """Render the N highest-norm (unnormalized) descriptor systems.

    Useful for diagnosing whether descriptor outliers (large L2 norm) correspond
    to chemically unusual systems or numerical edge cases. Norms are always
    computed on ``descriptors_raw`` regardless of the runner's normalization.
    """

    kind: Literal["top_norm_descriptors"] = "top_norm_descriptors"
    n_top: int = 10
    n_cols: int = 5
    figure_file_name: str = "top_norm_descriptors.png"
    summary_file_name: str = "top_norm_descriptors.yaml"
    rotation: str = "30x,30y,0z"
    panel_size: tuple[float, float] = (3.0, 3.0)

    def run(self, ctx: DescriptorAnalysisContext) -> list[EvalResult]:
        norms = np.linalg.norm(ctx.descriptors_raw, axis=1)
        n_top = min(self.n_top, norms.shape[0])
        top_idx = np.argsort(norms)[::-1][:n_top].tolist()
        top_atoms = _atoms_for_indices(ctx.dataset, top_idx)

        n_cols = max(1, min(self.n_cols, n_top))
        n_rows = (n_top + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(self.panel_size[0] * n_cols, self.panel_size[1] * n_rows),
            squeeze=False,
        )

        struct_ids = np.asarray(ctx.dataset.structure_ids[: ctx.dataset.N_structures])
        entries: list[_TopNormEntry] = []

        for panel, (idx, atoms) in enumerate(zip(top_idx, top_atoms, strict=True)):
            ax = axes[panel // n_cols][panel % n_cols]
            plot_atoms(atoms, ax=ax, rotation=self.rotation)
            ax.set_title(
                f"#{panel + 1} idx={idx} N={len(atoms)}\n‖z‖={norms[idx]:.2f}",
                fontsize=9,
            )
            entries.append(
                _TopNormEntry(
                    rank=panel + 1,
                    dataset_index=int(idx),
                    structure_id=int(struct_ids[idx]),
                    n_atoms=int(len(atoms)),
                    descriptor_norm=float(norms[idx]),
                    total_charge=float(atoms.info.get("total_charge", 0.0)),
                    total_spin=float(atoms.info.get("total_spin", 0.0)),
                )
            )

        for blank in range(n_top, n_rows * n_cols):
            axes[blank // n_cols][blank % n_cols].axis("off")

        fig.suptitle(f"Top {n_top} highest-norm descriptors")
        fig.tight_layout()

        return [
            FigureResult(
                file_name=ctx.file_name(self.figure_file_name), figure=fig
            ),
            PydanticResult(
                file_name=ctx.file_name(self.summary_file_name),
                obj=_TopNormSummary(entries=entries),
            ),
        ]


class ProjectionPlotTask(_BaseAnalysisTask):
    """2D scatter of the cached projection, colored by an arbitrary provider."""

    kind: Literal["projection_plot"] = "projection_plot"
    file_name: str
    color_provider: ColorProvider
    title: str | None = None
    point_size: float = 0.4
    alpha: float = 0.7
    figsize: tuple[float, float] = (8, 6)
    xlabel: str = "Component 1"
    ylabel: str = "Component 2"

    def run(self, ctx: DescriptorAnalysisContext) -> list[EvalResult]:
        if ctx.projection is None:
            raise ValueError(
                "ProjectionPlotTask requires a cached 2D projection on the context "
                "(set projection_config on the runner)."
            )

        spec = self.color_provider.compute(ctx)

        fig, ax = plt.subplots(figsize=self.figsize)
        scatter = ax.scatter(
            ctx.projection[:, 0],
            ctx.projection[:, 1],
            c=spec.values,
            s=self.point_size,
            alpha=self.alpha,
            **spec.scatter_kwargs,
        )
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        title = self.title or self.color_provider.title_suffix()
        ax.set_title(title)

        if spec.colorbar:
            cbar = fig.colorbar(scatter, ax=ax)
            if spec.colorbar_label is not None:
                cbar.set_label(spec.colorbar_label)
        if spec.legend_handles:
            ax.legend(handles=spec.legend_handles, fontsize="small")

        fig.tight_layout()
        return [FigureResult(file_name=ctx.file_name(self.file_name), figure=fig)]


class ChemiscopeProjectionTask(_BaseAnalysisTask):
    """Bundles the projection + dataset frames into a chemiscope viewer file."""

    kind: Literal["chemiscope_projection"] = "chemiscope_projection"
    file_name: str = "chemiscope_projection.json.gz"
    component_label: str = "PC"

    def run(self, ctx: DescriptorAnalysisContext) -> list[EvalResult]:
        if ctx.projection is None:
            raise ValueError(
                "ChemiscopeProjectionTask requires a cached 2D projection."
            )

        import chemiscope

        properties = {self.component_label: ctx.projection}
        settings = chemiscope.quick_settings(
            x=f"{self.component_label}[1]", y=f"{self.component_label}[2]"
        )
        frames = ctx.dataset.get_all_molecules()
        data = chemiscope.create_input(
            frames=frames, properties=properties, settings=settings
        )
        return [ChemiscopeResult(file_name=ctx.file_name(self.file_name), data=data)]


class HDBSCANClusterTask(_BaseAnalysisTask):
    """Cluster descriptors via UMAP→HDBSCAN; plot using the cached 2D projection.

    Runs a separate higher-dimensional UMAP tuned for clustering (low ``min_dist``,
    cosine metric by default), then HDBSCAN on that embedding. Cluster labels are
    overlaid on ``ctx.projection``; HDBSCAN noise (``label == -1``) is drawn in gray.
    """

    kind: Literal["hdbscan_cluster"] = "hdbscan_cluster"

    cluster_n_components: int = 15
    cluster_n_neighbors: int = 30
    cluster_min_dist: float = 0.0
    cluster_metric: str = "cosine"

    min_cluster_size: int = 50
    min_samples: int | None = None
    cluster_selection_epsilon: float = 0.0

    random_state: int | None = None

    figure_file_name: str = "hdbscan_clusters.png"
    summary_file_name: str = "hdbscan_clusters.yaml"

    point_size: float = 0.4
    alpha: float = 0.7
    figsize: tuple[float, float] = (8, 6)

    def run(self, ctx: DescriptorAnalysisContext) -> list[EvalResult]:
        if ctx.projection is None:
            raise ValueError(
                "HDBSCANClusterTask requires a cached 2D projection on the context "
                "(set projection_config on the runner)."
            )

        import umap
        from sklearn.cluster import HDBSCAN

        cache_key = (
            "hdbscan_cluster_umap",
            self.cluster_n_components,
            self.cluster_n_neighbors,
            self.cluster_min_dist,
            self.cluster_metric,
            self.random_state,
        )
        cluster_embedding = ctx.cache.get(cache_key)
        if cluster_embedding is None:
            cluster_embedding = umap.UMAP(
                n_components=self.cluster_n_components,
                n_neighbors=self.cluster_n_neighbors,
                min_dist=self.cluster_min_dist,
                metric=self.cluster_metric,
                random_state=self.random_state,
            ).fit_transform(ctx.descriptors)
            ctx.cache[cache_key] = cluster_embedding

        labels = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
        ).fit_predict(cluster_embedding)

        cluster_ids = np.unique(labels[labels >= 0])
        n_clusters = int(cluster_ids.size)
        noise_mask = labels == -1
        n_noise = int(noise_mask.sum())
        n_points = int(labels.shape[0])

        fig, ax = plt.subplots(figsize=self.figsize)
        if n_noise > 0:
            ax.scatter(
                ctx.projection[noise_mask, 0],
                ctx.projection[noise_mask, 1],
                c="lightgray",
                s=self.point_size,
                alpha=self.alpha,
                label=f"noise (n={n_noise})",
            )

        cmap = plt.get_cmap("tab20", max(1, n_clusters))
        for i, cid in enumerate(cluster_ids.tolist()):
            mask = labels == cid
            ax.scatter(
                ctx.projection[mask, 0],
                ctx.projection[mask, 1],
                color=cmap(i),
                s=self.point_size,
                alpha=self.alpha,
                label=f"{cid} (n={int(mask.sum())})",
            )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(
            f"HDBSCAN on {self.cluster_n_components}D UMAP: "
            f"{n_clusters} clusters, {n_noise}/{n_points} noise"
        )
        if 1 <= n_clusters <= 20:
            ax.legend(fontsize="x-small", markerscale=4, ncol=2, loc="best")
        fig.tight_layout()

        summary = _HDBSCANClusterSummary(
            n_points=n_points,
            n_clusters=n_clusters,
            n_noise=n_noise,
            noise_fraction=float(n_noise) / float(n_points) if n_points else 0.0,
            cluster_sizes={
                int(cid): int(np.sum(labels == cid))
                for cid in cluster_ids.tolist()
            },
            cluster_n_components=self.cluster_n_components,
            cluster_n_neighbors=self.cluster_n_neighbors,
            cluster_min_dist=self.cluster_min_dist,
            cluster_metric=self.cluster_metric,
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
        )

        return [
            FigureResult(file_name=ctx.file_name(self.figure_file_name), figure=fig),
            PydanticResult(
                file_name=ctx.file_name(self.summary_file_name), obj=summary
            ),
        ]


DescriptorAnalysisTask = Annotated[
    CapacityDiagnosticTask
    | DescriptorDistributionTask
    | TopNormDescriptorsTask
    | ProjectionPlotTask
    | ChemiscopeProjectionTask
    | HDBSCANClusterTask,
    Field(discriminator="kind"),
]


class _DescriptorDistributionSummary(BaseModel):
    """Summary statistics emitted alongside the distribution plots."""

    n_samples: int
    d_latent: int
    value_mean: float
    value_std: float
    value_min: float
    value_max: float
    norm_mean: float
    norm_std: float
    norm_min: float
    norm_max: float


class _TopNormEntry(BaseModel):
    rank: int
    dataset_index: int
    structure_id: int
    n_atoms: int
    descriptor_norm: float
    total_charge: float
    total_spin: float


class _TopNormSummary(BaseModel):
    entries: list[_TopNormEntry]


class _HDBSCANClusterSummary(BaseModel):
    n_points: int
    n_clusters: int
    n_noise: int
    noise_fraction: float
    cluster_sizes: dict[int, int]
    cluster_n_components: int
    cluster_n_neighbors: int
    cluster_min_dist: float
    cluster_metric: str
    min_cluster_size: int
    min_samples: int | None
    cluster_selection_epsilon: float


def _atoms_for_indices(
    dataset: MoleculeDataset, indices: list[int]
) -> list[Atoms]:
    """Materialize ASE Atoms only for the requested structure indices."""
    if not indices:
        return []
    n_struct = dataset.N_structures
    ptr = np.asarray(dataset.ptr[: n_struct + 1], dtype=np.int64)
    total_charge = np.asarray(dataset.total_charge[:n_struct])
    total_spin = np.asarray(dataset.total_spin[:n_struct])

    out: list[Atoms] = []
    for idx in indices:
        a0, a1 = int(ptr[idx]), int(ptr[idx + 1])
        numbers = np.asarray(dataset.atomic_numbers[a0:a1], dtype=np.int64)
        positions = np.asarray(dataset.positions[a0:a1], dtype=np.float32)
        out.append(
            Atoms(
                numbers=numbers,
                positions=positions,
                info={
                    "total_charge": float(total_charge[idx]),
                    "total_spin": float(total_spin[idx]),
                },
            )
        )
    return out


def _plot_per_dim_entropy(report: LatentCapacityReportModel) -> Any:
    fig, ax = plt.subplots(figsize=(10, 4))
    order = np.argsort(report.H_norm_per_dim)[::-1]
    ax.bar(np.arange(len(order)), report.H_norm_per_dim[order], width=1.0)
    ax.axhline(0.2, color="red", linestyle="--", linewidth=1, label="dead threshold")
    ax.set_xlabel("Latent dimension (sorted)")
    ax.set_ylabel("H_i / log2(bins_i)")
    title = (
        f"Per-dim entropy utilisation "
        f"(d_eff={report.d_eff:.1f}, dead={report.dead_dims}, util={report.utilisation:.2f})"
    )
    if report.n_dropped:
        title += (
            f"\ndropped {report.n_dropped} outliers "
            f"(‖z‖ > {report.norm_threshold:.2f})"
        )
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_explained_variance(report: LatentCapacityReportModel) -> Any:
    fig, ax = plt.subplots(figsize=(8, 4))
    cumulative = np.cumsum(report.explained_var_ratio)
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker=".")
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.set_title("Cumulative explained variance of descriptor covariance")
    fig.tight_layout()
    return fig
