from __future__ import annotations

import math
import re
from abc import abstractmethod
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal, cast

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


def compute_hdbscan_labels(
    ctx: DescriptorAnalysisContext,
    *,
    reducer: Literal["umap", "none"],
    cluster_n_components: int,
    cluster_n_neighbors: int,
    cluster_min_dist: float,
    cluster_metric: str,
    random_state: int | None,
    min_cluster_size: int,
    min_samples: int | None,
    cluster_selection_epsilon: float,
    use_raw_descriptors: bool = False,
) -> np.ndarray:
    """HDBSCAN cluster labels (-1 = noise) on the descriptors.

    ``reducer="umap"`` clusters a higher-dimensional clustering UMAP (cached on
    the context by its parameters, so the figures / chemiscope / fingerprints
    share one embedding). ``reducer="none"`` runs HDBSCAN directly on the
    descriptors with the Euclidean metric.

    ``use_raw_descriptors=True`` clusters the *unnormalized* descriptors so the
    vector norms carry signal — meaningful only with ``cluster_metric=
    "euclidean"`` (cosine and L2-normalized euclidean would both ignore the
    norm, making them equivalent to the canonical cosine path). The default
    ``False`` uses the runner's normalized descriptors so the canonical
    cosine-on-unit-vectors protocol is unchanged.
    """
    from sklearn.cluster import HDBSCAN

    descriptors = ctx.descriptors_raw if use_raw_descriptors else ctx.descriptors

    # Cache the fit labels per (reducer, mcs, *params) so the sweep and the
    # fingerprint at the same setting don't refit HDBSCAN twice — that doubled
    # the no-UMAP cost and burned the slurm time budget on the previous run.
    # ``use_raw_descriptors`` is in the key so the canonical and the
    # euclidean-on-raw variants don't collide.
    labels_key = (
        f"hdbscan_labels:{reducer}:{cluster_n_components}:{cluster_n_neighbors}:"
        f"{cluster_min_dist}:{cluster_metric}:{random_state}:{min_cluster_size}:"
        f"{min_samples}:{cluster_selection_epsilon}:raw={use_raw_descriptors}"
    )
    cached_labels = ctx.cache.get(labels_key)
    if cached_labels is not None:
        return cast("np.ndarray", cached_labels)

    if reducer == "none":
        # 108k x 64 floats → ball_tree handles this near O(n log n);
        # auto-selection at D=64 can fall back to brute O(n^2).
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric="euclidean",
            algorithm="ball_tree",
        ).fit_predict(np.ascontiguousarray(descriptors))
        ctx.cache[labels_key] = labels
        return labels

    import umap

    cache_key = (
        f"hdbscan_cluster_umap:{cluster_n_components}:{cluster_n_neighbors}:"
        f"{cluster_min_dist}:{cluster_metric}:{random_state}:raw={use_raw_descriptors}"
    )
    cluster_embedding = ctx.cache.get(cache_key)
    if cluster_embedding is None:
        cluster_embedding = umap.UMAP(
            n_components=cluster_n_components,
            n_neighbors=cluster_n_neighbors,
            min_dist=cluster_min_dist,
            metric=cluster_metric,
            random_state=random_state,
        ).fit_transform(descriptors)
        ctx.cache[cache_key] = cluster_embedding

    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    ).fit_predict(cluster_embedding)
    ctx.cache[labels_key] = labels
    return labels


class _ClusterTaskBase(_BaseAnalysisTask):
    """Shared UMAP→HDBSCAN parameters for clustering-derived tasks."""

    cluster_reducer: Literal["umap", "none"] = "umap"
    cluster_n_components: int = 15
    cluster_n_neighbors: int = 30
    cluster_min_dist: float = 0.0
    cluster_metric: str = "cosine"
    use_raw_descriptors: bool = False
    """Cluster on ``ctx.descriptors_raw`` (pre-normalization) so the vector
    norm carries signal. Only meaningful when paired with
    ``cluster_metric="euclidean"`` — cosine and L2-normalized Euclidean both
    discard the norm, making them equivalent to the canonical path."""

    min_samples: int | None = None
    cluster_selection_epsilon: float = 0.0
    random_state: int | None = None

    def _labels(
        self, ctx: DescriptorAnalysisContext, min_cluster_size: int
    ) -> np.ndarray:
        return compute_hdbscan_labels(
            ctx,
            reducer=self.cluster_reducer,
            cluster_n_components=self.cluster_n_components,
            cluster_n_neighbors=self.cluster_n_neighbors,
            cluster_min_dist=self.cluster_min_dist,
            cluster_metric=self.cluster_metric,
            random_state=self.random_state,
            min_cluster_size=min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            use_raw_descriptors=self.use_raw_descriptors,
        )


def _metal_env_features(ctx: DescriptorAnalysisContext) -> list[Any]:
    """Cached per-structure chemical descriptors (one pymatgen pass)."""
    feats = ctx.cache.get("metal_env_features")
    if feats is None:
        from threedscriptors.evaluation.descriptor_analysis.tmqm_chemical_features import (
            compute_metal_environment_features,
        )

        feats = compute_metal_environment_features(ctx.dataset.get_all_molecules())
        ctx.cache["metal_env_features"] = feats
    return cast("list[Any]", feats)


def _cluster_label_str(label: int) -> str:
    return "noise" if label == -1 else f"c{label}"


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
        top_subset_idx = np.argsort(norms)[::-1][:n_top]
        top_dataset_idx = ctx.to_dataset_indices(top_subset_idx)
        top_atoms = _atoms_for_indices(ctx.dataset, top_dataset_idx.tolist())

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

        for panel, (subset_idx, dataset_idx, atoms) in enumerate(
            zip(
                top_subset_idx.tolist(), top_dataset_idx.tolist(), top_atoms,
                strict=True,
            )
        ):
            ax = axes[panel // n_cols][panel % n_cols]
            plot_atoms(atoms, ax=ax, rotation=self.rotation)
            ax.set_title(
                f"#{panel + 1} idx={dataset_idx} N={len(atoms)}\n"
                f"‖z‖={norms[subset_idx]:.2f}",
                fontsize=9,
            )
            entries.append(
                _TopNormEntry(
                    rank=panel + 1,
                    dataset_index=int(dataset_idx),
                    structure_id=int(struct_ids[dataset_idx]),
                    n_atoms=len(atoms),
                    descriptor_norm=float(norms[subset_idx]),
                    total_charge=float(atoms.info.get("total_charge", 0.0)),
                    multiplicity=float(atoms.info.get("multiplicity", 0.0)),
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


class HDBSCANClusterTask(_ClusterTaskBase):
    """Cluster descriptors via UMAP→HDBSCAN; plot using the cached 2D projection.

    Runs a separate higher-dimensional UMAP tuned for clustering (low ``min_dist``,
    cosine metric by default), then HDBSCAN on that embedding. Cluster labels are
    overlaid on ``ctx.projection``; HDBSCAN noise (``label == -1``) is drawn in gray.
    """

    kind: Literal["hdbscan_cluster"] = "hdbscan_cluster"

    min_cluster_size: int = 50

    figure_file_name: str = "hdbscan_clusters.png"
    summary_file_name: str = "hdbscan_clusters.yaml"

    point_size: float = 0.4
    alpha: float = 0.7
    figsize: tuple[float, float] = (8, 6)

    cmap_small: str = "tab20"
    cmap_large: str = "gist_ncar"
    shuffle_colors: bool = True
    color_shuffle_seed: int = 0

    def run(self, ctx: DescriptorAnalysisContext) -> list[EvalResult]:
        if ctx.projection is None:
            raise ValueError(
                "HDBSCANClusterTask requires a cached 2D projection on the context "
                "(set projection_config on the runner)."
            )

        labels = self._labels(ctx, self.min_cluster_size)

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

        if n_clusters <= 20:
            cmap = plt.get_cmap(self.cmap_small, max(1, n_clusters))
            cluster_colors = [cmap(i) for i in range(n_clusters)]
        else:
            # Sample evenly across a continuous spectrum so 200+ clusters stay
            # visually distinguishable; shuffle so adjacent IDs (often spatially
            # adjacent) get distant hues.
            cmap = plt.get_cmap(self.cmap_large)
            sample_points = np.linspace(0.02, 0.98, n_clusters)
            if self.shuffle_colors:
                rng = np.random.default_rng(self.color_shuffle_seed)
                sample_points = rng.permutation(sample_points)
            cluster_colors = [cmap(p) for p in sample_points]

        for i, cid in enumerate(cluster_ids.tolist()):
            mask = labels == cid
            ax.scatter(
                ctx.projection[mask, 0],
                ctx.projection[mask, 1],
                color=cluster_colors[i],
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


class ChemiscopeClusterTask(_ClusterTaskBase):
    """One chemiscope viewer for manual chemical inspection of clusters.

    Bundles, on the *settled* 2D UMAP map (``ctx.projection``): the 3D
    structures, every requested HDBSCAN granularity as a switchable categorical
    coloring, and per-structure chemical descriptors (metal, d-block,
    coordination number, first-sphere donor set, hapticity / sandwich &
    carborane flags, element counts, formula). In chemiscope you flip the map
    color between ``cluster_mcs*`` and any chemical property to see, by eye,
    whether a cluster is e.g. all phosphines, η6-arenes or diimine-octahedral.
    """

    kind: Literal["chemiscope_cluster"] = "chemiscope_cluster"

    min_cluster_sizes: list[int] = [50, 200, 1000, 5000]
    default_color_mcs: int = 200
    file_name: str = "chemiscope_clusters.json.gz"
    component_label: str = "UMAP"
    include_heavy_properties: bool = False
    """Attach the high-cardinality / per-element columns (donor_set, formula,
    n_B/P/O/N/S/halogen). Off by default: with ~10⁵ structures these bloat the
    JSON well past what the chemiscope web viewer loads comfortably."""

    def run(self, ctx: DescriptorAnalysisContext) -> list[EvalResult]:
        if ctx.projection is None:
            raise ValueError(
                "ChemiscopeClusterTask requires a cached 2D projection "
                "(set projection_config on the runner)."
            )

        import chemiscope

        molecules = ctx.dataset.get_all_molecules()
        feats = _metal_env_features(ctx)
        n = len(molecules)

        def col(values: list[Any], description: str) -> dict[str, Any]:
            return {
                "target": "structure",
                "values": values,
                "description": description,
            }

        properties: dict[str, Any] = {
            self.component_label: col(
                np.asarray(ctx.projection)[:n].tolist(),
                "Settled 2D UMAP of the molecular descriptors",
            )
        }

        for mcs in self.min_cluster_sizes:
            labels = self._labels(ctx, mcs)
            # Store as ints (not 'c{id}'/'noise' strings) so the chemiscope
            # viewer accepts them as a numeric color axis — its categorical
            # color picker is capped at a handful of unique values and would
            # silently drop a property with hundreds of cluster ids.
            properties[f"cluster_mcs{mcs}"] = col(
                [int(x) for x in labels],
                f"HDBSCAN cluster id (min_cluster_size={mcs}); -1 = noise",
            )

        properties.update(
            {
                "metal": col(
                    [f.metal_symbol for f in feats], "TM center element"
                ),
                "metal_block": col(
                    [f.metal_block for f in feats], "TM center sub-shell row"
                ),
                "coordination_number": col(
                    [f.coordination_number for f in feats],
                    "Metal first-sphere CN (-1 = unresolved)",
                ),
                "hapticity_max": col(
                    [f.hapticity_max for f in feats],
                    "Largest contiguous carbon-donor group (Cp=5, arene=6)",
                ),
                "n_pi_groups": col(
                    [f.n_pi_groups for f in feats],
                    "Distinct π carbon-donor groups (sandwich ⇒ 2)",
                ),
                "is_sandwich": col(
                    ["yes" if f.is_sandwich else "no" for f in feats],
                    "Two π carbon-donor groups with hapticity ≥ 5",
                ),
                "is_carborane": col(
                    ["yes" if f.is_carborane else "no" for f in feats],
                    "≥4 boron atoms (borane/carborane cage)",
                ),
                "geometry_class": col(
                    [f.geometry_class for f in feats],
                    "Coarse coordination geometry from donor-M-donor angles",
                ),
                "n_carbonyl": col(
                    [f.n_carbonyl for f in feats], "Terminal CO ligands"
                ),
                "n_donor_H": col(
                    [f.n_donor_H for f in feats], "Hydride donors (M-H)"
                ),
                "n_chelate_rings": col(
                    [f.n_chelate_rings for f in feats],
                    "Donor pairs connected by a short backbone (chelate rings)",
                ),
                "max_chelate_ring_size": col(
                    [f.max_chelate_ring_size for f in feats],
                    "Largest 4-6 chelate ring size; 0 if monodentate only",
                ),
                "radius_of_gyration": col(
                    [f.radius_of_gyration for f in feats], "Size proxy"
                ),
                "n_atoms": col([f.n_atoms for f in feats], "Atom count"),
            }
        )

        if self.include_heavy_properties:
            properties.update(
                {
                    "donor_set": col(
                        [f.donor_set for f in feats],
                        "Canonical first-sphere donor signature, e.g. N4O2",
                    ),
                    "formula": col([f.formula for f in feats], "Hill formula"),
                    "n_B": col([f.n_B for f in feats], "Boron count"),
                    "n_P": col([f.n_P for f in feats], "Phosphorus count"),
                    "n_O": col([f.n_O for f in feats], "Oxygen count"),
                    "n_N": col([f.n_N for f in feats], "Nitrogen count"),
                    "n_S": col([f.n_S for f in feats], "Sulfur count"),
                    "n_halogen": col(
                        [f.n_halogen for f in feats], "Halogen count"
                    ),
                }
            )

        color = f"cluster_mcs{self.default_color_mcs}"
        if color not in properties:
            color = f"cluster_mcs{self.min_cluster_sizes[0]}"
        settings = chemiscope.quick_settings(
            x=f"{self.component_label}[1]",
            y=f"{self.component_label}[2]",
            map_color=color,
        )
        data = chemiscope.create_input(
            structures=molecules, properties=properties, settings=settings
        )
        return [ChemiscopeResult(file_name=ctx.file_name(self.file_name), data=data)]


class ClusterChemicalFingerprintTask(_ClusterTaskBase):
    """Per-cluster chemical summary that proposes a human label.

    Aggregates the metal-environment features within each HDBSCAN cluster
    (modal donor set, dominant metal block, CN, π / carborane fractions) and
    applies a rule heuristic to suggest a chemist-readable label
    (oxygen-donor, phosphine, sandwich, N-donor octahedral, carborane, …).
    Use it to triage which clusters to open in the chemiscope viewer.
    """

    kind: Literal["cluster_chemical_fingerprint"] = "cluster_chemical_fingerprint"

    min_cluster_size: int = 200
    top_k: int = 5
    summary_file_name: str = "cluster_chemical_fingerprint.yaml"

    def run(self, ctx: DescriptorAnalysisContext) -> list[EvalResult]:
        labels = self._labels(ctx, self.min_cluster_size)
        feats = _metal_env_features(ctx)

        order = sorted(
            np.unique(labels).tolist(),
            key=lambda cid: (cid == -1, -int(np.sum(labels == cid))),
        )
        fingerprints = [
            _fingerprint_cluster(
                int(cid),
                [feats[i] for i in np.flatnonzero(labels == cid).tolist()],
                self.top_k,
            )
            for cid in order
        ]

        mean_purity, mean_entropy, n_clusters, _ = _purity_for_labels(labels, feats)
        report = _ClusterFingerprintReport(
            reducer=self.cluster_reducer,
            min_cluster_size=self.min_cluster_size,
            n_points=int(labels.shape[0]),
            n_clusters=n_clusters,
            n_noise=int(np.sum(labels == -1)),
            mean_donor_set_purity=mean_purity,
            mean_donor_set_entropy_bits=mean_entropy,
            clusters=fingerprints,
        )
        return [
            PydanticResult(
                file_name=ctx.file_name(self.summary_file_name), obj=report
            )
        ]


class ClusterGranularitySweepTask(_ClusterTaskBase):
    """Compare HDBSCAN granularities by size-weighted donor-set purity.

    Runs HDBSCAN at several ``min_cluster_size`` values on the shared cached
    embedding and reports, per granularity, the size-weighted mean modal
    donor-set purity, entropy, cluster count and noise fraction — then flags
    the granularity that maximizes purity (tie-break: less noise, more
    clusters). This answers "at what resolution do the descriptor clusters
    become chemically coherent?" without eyeballing.
    """

    kind: Literal["cluster_granularity_sweep"] = "cluster_granularity_sweep"

    min_cluster_sizes: list[int] = [25, 50, 100]
    summary_file_name: str = "cluster_granularity_sweep.yaml"

    def run(self, ctx: DescriptorAnalysisContext) -> list[EvalResult]:
        feats = _metal_env_features(ctx)

        n_points = 0
        rows: list[_GranularityRow] = []
        for mcs in sorted(self.min_cluster_sizes):
            labels = self._labels(ctx, mcs)
            n_points = int(labels.shape[0])
            purity, entropy, n_clusters, noise_fraction = _purity_for_labels(
                labels, feats
            )
            rows.append(
                _GranularityRow(
                    min_cluster_size=mcs,
                    n_clusters=n_clusters,
                    noise_fraction=noise_fraction,
                    mean_donor_set_purity=purity,
                    mean_donor_set_entropy_bits=entropy,
                )
            )

        best = max(
            rows,
            key=lambda r: (
                r.mean_donor_set_purity,
                -r.noise_fraction,
                r.n_clusters,
            ),
        )
        best.recommended = True

        report = _ClusterGranularitySweepReport(
            reducer=self.cluster_reducer,
            n_points=n_points,
            recommended_min_cluster_size=best.min_cluster_size,
            rows=rows,
        )
        return [
            PydanticResult(
                file_name=ctx.file_name(self.summary_file_name), obj=report
            )
        ]


class ClusterAxisAnalysisTask(_ClusterTaskBase):
    """For each cluster: which chemical axis best explains it?

    Per-cluster, evaluates the modal-category fraction along every candidate
    axis (metal, block, coordination number, geometry class, dominant donor
    element, donor set, ligand motif family, size band) and reports the
    axis-with-highest-purity. Dataset-wide it aggregates (a) the size-weighted
    mean purity per axis and (b) how often each axis is the best explanation
    — directly answering "what *is* the right interpretation for these
    clusters" rather than fixing on donor-set alone.
    """

    kind: Literal["cluster_axis_analysis"] = "cluster_axis_analysis"

    min_cluster_size: int = 25
    summary_file_name: str = "cluster_axis_analysis.yaml"

    def run(self, ctx: DescriptorAnalysisContext) -> list[EvalResult]:
        labels = self._labels(ctx, self.min_cluster_size)
        feats = _metal_env_features(ctx)

        cluster_ids = sorted(
            np.unique(labels[labels >= 0]).tolist(),
            key=lambda cid: -int(np.sum(labels == cid)),
        )

        rows: list[_ClusterAxisRow] = []
        axis_size_purity: dict[str, float] = {a: 0.0 for a in _AXES}
        axis_wins: dict[str, int] = dict.fromkeys(_AXES, 0)
        total_size = 0

        for cid in cluster_ids:
            members = [feats[i] for i in np.flatnonzero(labels == cid).tolist()]
            size = len(members)
            total_size += size

            per_axis_purity: dict[str, float] = {}
            per_axis_modal: dict[str, str] = {}
            for axis_name, extractor in _AXES.items():
                modal, frac = _axis_modal_purity(members, extractor)
                per_axis_purity[axis_name] = frac
                per_axis_modal[axis_name] = modal
                axis_size_purity[axis_name] += size * frac

            best_axis = max(per_axis_purity, key=per_axis_purity.__getitem__)
            axis_wins[best_axis] += 1

            rows.append(
                _ClusterAxisRow(
                    cluster_id=int(cid),
                    size=size,
                    best_axis=best_axis,
                    best_category=per_axis_modal[best_axis],
                    best_purity=per_axis_purity[best_axis],
                    purities=per_axis_purity,
                    modal_categories=per_axis_modal,
                )
            )

        axis_summaries = sorted(
            (
                _AxisSummary(
                    name=name,
                    weighted_mean_purity=(
                        axis_size_purity[name] / total_size if total_size else 0.0
                    ),
                    n_clusters_won=axis_wins[name],
                )
                for name in _AXES
            ),
            key=lambda s: -s.weighted_mean_purity,
        )

        report = _ClusterAxisAnalysisReport(
            reducer=self.cluster_reducer,
            min_cluster_size=self.min_cluster_size,
            n_points=int(labels.shape[0]),
            n_clusters=len(cluster_ids),
            n_noise=int(np.sum(labels == -1)),
            axes=axis_summaries,
            clusters=rows,
        )
        return [
            PydanticResult(
                file_name=ctx.file_name(self.summary_file_name), obj=report
            )
        ]


class DescriptorStructureBenchmarkTask(_ClusterTaskBase):
    """Single-YAML benchmark scorecard of how chemically structured a descriptor is.

    Fixes the clustering protocol (canonical UMAP-15D cosine HDBSCAN at mcs=25,
    random_state=0) and reports, per chemical axis, the **Adjusted Mutual
    Information** between the HDBSCAN partition and that axis. AMI is
    chance-corrected, so axes with very different cardinalities (donor_set has
    thousands of categories, size_band has three) are directly comparable.

    The headline aggregate (``aggregate_chemistry_ami``) averages AMI across
    the chemistry-meaningful axes only (donor_set, ligand_motif,
    geometry_class, dominant_donor_element) — explicitly excluding the
    "free-purity" coarse axes (size_band, metal_block) so a descriptor that
    only encodes metal row + size cannot game the benchmark by trivial
    organization. A sweep over neighbouring ``min_cluster_size`` values is
    included for transparency, but the canonical scorecard fixes mcs=25.

    Designed to be a per-model artifact: one ``descriptor_structure_benchmark.yaml``
    per training run, comparable across models when the same dataset and
    protocol are used. See the docstring of the YAML's ``protocol`` field for
    the protocol version that must match across benchmark entries.
    """

    kind: Literal["descriptor_structure_benchmark"] = (
        "descriptor_structure_benchmark"
    )

    canonical_min_cluster_size: int = 25
    sweep_min_cluster_sizes: list[int] = [25, 50, 100]
    chemistry_axes: list[str] = [
        "donor_set",
        "ligand_motif",
        "geometry_class",
        "dominant_donor_element",
    ]
    pure_island_min_size: int = 50
    pure_island_min_purity: float = 0.9
    # Pinning the seed makes the score reproducible across re-runs of the same
    # descriptors. Override the base default of None.
    random_state: int | None = 0
    summary_file_name: str = "descriptor_structure_benchmark.yaml"

    def run(self, ctx: DescriptorAnalysisContext) -> list[EvalResult]:
        from sklearn.metrics import adjusted_mutual_info_score

        feats = _metal_env_features(ctx)
        # Per-axis category vector covering every structure (axis values are
        # constant in the descriptor, so we compute them once and reuse).
        axis_values = {
            name: [extractor(f) for f in feats] for name, extractor in _AXES.items()
        }

        mcs_values = sorted({*self.sweep_min_cluster_sizes, self.canonical_min_cluster_size})
        sweep: list[_BenchmarkSweepRow] = []
        canonical_labels: np.ndarray | None = None
        n_points = 0
        for mcs in mcs_values:
            labels = self._labels(ctx, mcs)
            n_points = int(labels.shape[0])
            ami_per_axis = {
                name: float(adjusted_mutual_info_score(values, labels))
                for name, values in axis_values.items()
            }
            cluster_ids = np.unique(labels[labels >= 0])
            sweep.append(
                _BenchmarkSweepRow(
                    min_cluster_size=mcs,
                    n_clusters=int(cluster_ids.size),
                    noise_fraction=float(np.sum(labels == -1)) / n_points if n_points else 0.0,
                    ami_per_axis=ami_per_axis,
                )
            )
            if mcs == self.canonical_min_cluster_size:
                canonical_labels = labels

        assert canonical_labels is not None  # mcs_values always contains canonical
        canonical = next(
            r for r in sweep if r.min_cluster_size == self.canonical_min_cluster_size
        )

        # Verify the chemistry-axis selection matches axes we actually computed
        # — guards against silent typos in the config.
        for ax in self.chemistry_axes:
            if ax not in canonical.ami_per_axis:
                raise ValueError(
                    f"chemistry_axes entry {ax!r} is not a known axis; "
                    f"available: {sorted(canonical.ami_per_axis)}"
                )
        aggregate_chemistry_ami = float(
            np.mean([canonical.ami_per_axis[a] for a in self.chemistry_axes])
        )

        # Pure-island tally at the canonical mcs: clusters big enough and pure
        # enough on at least one axis to be a recognisable chemical family.
        islands_by_axis: dict[str, int] = dict.fromkeys(_AXES, 0)
        total_islands = 0
        for cid in np.unique(canonical_labels[canonical_labels >= 0]).tolist():
            members = [
                feats[i] for i in np.flatnonzero(canonical_labels == cid).tolist()
            ]
            if len(members) < self.pure_island_min_size:
                continue
            best_axis: str | None = None
            best_purity = 0.0
            for axis_name, extractor in _AXES.items():
                _, frac = _axis_modal_purity(members, extractor)
                if frac > best_purity:
                    best_purity = frac
                    best_axis = axis_name
            if best_axis is not None and best_purity >= self.pure_island_min_purity:
                islands_by_axis[best_axis] += 1
                total_islands += 1

        # Resolve a reproducible dataset identifier. ``Path`` keeps both the
        # absolute path (for our records) and the basename (for human reading).
        dataset_path = str(getattr(ctx.dataset, "path", "unknown"))

        report = _DescriptorStructureBenchmarkReport(
            protocol=_BenchmarkProtocol(
                version=_BENCHMARK_PROTOCOL_VERSION,
                dataset_path=dataset_path,
                reducer=self.cluster_reducer,
                cluster_n_components=self.cluster_n_components,
                cluster_n_neighbors=self.cluster_n_neighbors,
                cluster_min_dist=self.cluster_min_dist,
                cluster_metric=self.cluster_metric,
                use_raw_descriptors=self.use_raw_descriptors,
                random_state=self.random_state,
                canonical_min_cluster_size=self.canonical_min_cluster_size,
                chemistry_axes=list(self.chemistry_axes),
                pure_island_min_size=self.pure_island_min_size,
                pure_island_min_purity=self.pure_island_min_purity,
            ),
            n_points=n_points,
            aggregate_chemistry_ami=aggregate_chemistry_ami,
            canonical=canonical,
            n_pure_islands={
                "total": total_islands,
                **{k: v for k, v in islands_by_axis.items() if v > 0},
            },
            sweep=sweep,
        )
        return [
            PydanticResult(
                file_name=ctx.file_name(self.summary_file_name), obj=report
            )
        ]


DescriptorAnalysisTask = Annotated[
    CapacityDiagnosticTask
    | DescriptorDistributionTask
    | TopNormDescriptorsTask
    | ProjectionPlotTask
    | ChemiscopeProjectionTask
    | ChemiscopeClusterTask
    | ClusterChemicalFingerprintTask
    | ClusterGranularitySweepTask
    | ClusterAxisAnalysisTask
    | DescriptorStructureBenchmarkTask
    | HDBSCANClusterTask,
    Field(discriminator="kind"),
]


# Bump this when the protocol (axes, mcs, normalization, …) changes so that
# old scorecards aren't silently compared against new ones.
_BENCHMARK_PROTOCOL_VERSION = "1.0"


def _dominant_donor_element(feat: Any) -> str:
    counts = {k: v for k, v in feat.donor_counts.items() if k != "C"}
    if not counts:
        return "C" if feat.donor_counts.get("C", 0) > 0 else "unresolved"
    return max(counts, key=counts.__getitem__)


def _ligand_motif(feat: Any) -> str:
    """Coarse single-token ligand-family label per structure.

    Rules are evaluated in order; the first match wins. The ordering is
    deliberate: the more *specific* / structural motifs (carborane cage,
    sandwich/Cp/arene) outrank the simpler donor-element heuristics so a
    sandwich complex with halide co-ligands is still labeled "sandwich".
    """
    rules = (
        (feat.is_carborane, "carborane"),
        (feat.is_sandwich, "sandwich"),
        (feat.hapticity_max == 6, "arene-like"),
        (feat.hapticity_max == 5, "Cp-like"),
        (feat.n_carbonyl >= 3, "carbonyl-rich"),
        (feat.n_donor_H >= 1, "hydride-bearing"),
        (feat.n_donor_halogen >= 2, "halide-rich"),
        (feat.n_donor_P >= 2, "phosphine-rich"),
        (feat.n_chelate_rings >= 2, "chelate-multidentate"),
        (feat.n_chelate_rings == 1, "chelate-bidentate"),
    )
    for predicate, label in rules:
        if predicate:
            return label
    return "other"


def _size_band(feat: Any) -> str:
    n = feat.n_atoms
    if n < 25:
        return "small"
    if n < 60:
        return "medium"
    return "large"


def _cn_bucket(feat: Any) -> str:
    return f"CN{feat.coordination_number}" if feat.coordination_number >= 0 else "?"


# Candidate chemical axes the cluster partition might be explaining. Ordered so
# the per-cluster best-axis tie-breaking falls back on the simplest description
# (coarser categories first).
_AXES: dict[str, Any] = {
    "metal_block": lambda f: f.metal_block,
    "size_band": _size_band,
    "ligand_motif": _ligand_motif,
    "geometry_class": lambda f: f.geometry_class,
    "coordination_number": _cn_bucket,
    "dominant_donor_element": _dominant_donor_element,
    "metal_element": lambda f: f.metal_symbol,
    "donor_set": lambda f: f.donor_set or "unresolved",
}


def _axis_modal_purity(members: list[Any], extractor: Any) -> tuple[str, float]:
    values = [extractor(m) for m in members]
    counter = Counter(values)
    modal, n = counter.most_common(1)[0]
    return str(modal), n / len(values)


def _top_counts(items: list[Any], k: int) -> dict[str, int]:
    return {str(key): int(c) for key, c in Counter(items).most_common(k)}


_DONOR_TOKEN = re.compile(r"([A-Z][a-z]?)(\d+)")


def _parse_donor_set(label: str) -> dict[str, int]:
    return {el: int(n) for el, n in _DONOR_TOKEN.findall(label)}


def _modal_purity_entropy(members: list[Any]) -> tuple[str, float, float]:
    """(modal donor-set, its fraction, Shannon entropy in bits).

    Computed over members with a *resolved* donor set. Purity (modal fraction)
    near 1 and entropy near 0 ⇒ the cluster is one donor motif; high entropy ⇒
    the cluster is a chemical mixture. The honest coherence signal.
    """
    sets = [m.donor_set for m in members if m.donor_set]
    if not sets:
        return "", 0.0, 0.0
    counts = Counter(sets)
    n = len(sets)
    modal, modal_n = counts.most_common(1)[0]
    entropy = -sum((v / n) * math.log2(v / n) for v in counts.values())
    return modal, modal_n / n, entropy


def _purity_for_labels(
    labels: np.ndarray, feats: list[Any]
) -> tuple[float, float, int, float]:
    """Size-weighted mean donor-set purity/entropy over non-noise clusters."""
    cluster_ids = np.unique(labels[labels >= 0])
    n_points = int(labels.shape[0])
    noise_fraction = float(np.sum(labels == -1)) / n_points if n_points else 0.0

    total_w = 0
    w_purity = 0.0
    w_entropy = 0.0
    for cid in cluster_ids.tolist():
        members = [feats[i] for i in np.flatnonzero(labels == cid).tolist()]
        _, frac, entropy = _modal_purity_entropy(members)
        w = len(members)
        total_w += w
        w_purity += w * frac
        w_entropy += w * entropy

    mean_purity = w_purity / total_w if total_w else 0.0
    mean_entropy = w_entropy / total_w if total_w else 0.0
    return mean_purity, mean_entropy, int(cluster_ids.size), noise_fraction


def _fingerprint_cluster(
    cluster_id: int, members: list[Any], top_k: int
) -> _ClusterFingerprint:
    size = len(members)
    resolved_cn = [m.coordination_number for m in members if m.coordination_number >= 0]
    modal_cn = Counter(resolved_cn).most_common(1)[0][0] if resolved_cn else -1

    modal_set, modal_frac, donor_entropy = _modal_purity_entropy(members)

    frac_sandwich = sum(m.is_sandwich for m in members) / size if size else 0.0
    frac_carborane = sum(m.is_carborane for m in members) / size if size else 0.0
    mean_hapticity = float(np.mean([m.hapticity_max for m in members])) if size else 0.0

    return _ClusterFingerprint(
        cluster_id=cluster_id,
        size=size,
        suggested_label=_suggest_label(
            modal_set, modal_frac, int(modal_cn), frac_sandwich,
            frac_carborane, mean_hapticity,
        ),
        modal_donor_set=modal_set,
        modal_donor_set_frac=modal_frac,
        donor_set_entropy_bits=donor_entropy,
        top_metals=_top_counts([m.metal_symbol for m in members], top_k),
        metal_blocks=_top_counts([m.metal_block for m in members], top_k),
        top_donor_sets=_top_counts(
            [m.donor_set for m in members if m.donor_set], top_k
        ),
        modal_coordination_number=int(modal_cn),
        mean_coordination_number=(
            float(np.mean(resolved_cn)) if resolved_cn else -1.0
        ),
        mean_hapticity_max=mean_hapticity,
        frac_sandwich=frac_sandwich,
        frac_carborane=frac_carborane,
        mean_n_P=float(np.mean([m.n_P for m in members])) if size else 0.0,
        mean_n_O=float(np.mean([m.n_O for m in members])) if size else 0.0,
        mean_n_N=float(np.mean([m.n_N for m in members])) if size else 0.0,
        mean_n_B=float(np.mean([m.n_B for m in members])) if size else 0.0,
    )


_DONOR_LABELS = {
    "P": "phosphine / P-donor",
    "O": "oxygen-donor",
    "S": "thiolate / S-donor",
    "F": "halide",
    "Cl": "halide",
    "Br": "halide",
    "I": "halide",
}

# A cluster only gets a chemical label if a strict majority of its members
# share one donor signature; otherwise it is reported as a mixture rather than
# mislabeled from the most-frequent element (the old summed-counts bug).
_PURE_LABEL_MIN_FRAC = 0.50


def _suggest_label(
    modal_set: str,
    modal_frac: float,
    modal_cn: int,
    frac_sandwich: float,
    frac_carborane: float,
    mean_hapticity: float,
) -> str:
    if frac_carborane >= 0.5:
        return "carborane / borane-cage"
    if frac_sandwich >= 0.5:
        return "sandwich (bis-π)"
    if mean_hapticity >= 4.5:
        return "η5/η6 (Cp / arene)"
    if not modal_set or modal_frac < _PURE_LABEL_MIN_FRAC:
        return f"mixed (modal {modal_set or 'unresolved'} {modal_frac:.0%})"
    non_c = _parse_donor_set(modal_set)
    non_c.pop("C", None)
    if not non_c:
        base = "sigma-carbon / carbonyl-like"
    else:
        dominant = max(non_c, key=non_c.__getitem__)
        if dominant == "N":
            base = "N-donor octahedral (diimine?)" if modal_cn == 6 else "N-donor"
        else:
            base = _DONOR_LABELS.get(dominant, f"{dominant}-donor")
    return f"{base} [{modal_set} {modal_frac:.0%}]"


class _ClusterFingerprint(BaseModel):
    cluster_id: int
    size: int
    suggested_label: str
    modal_donor_set: str
    modal_donor_set_frac: float
    donor_set_entropy_bits: float
    top_metals: dict[str, int]
    metal_blocks: dict[str, int]
    top_donor_sets: dict[str, int]
    modal_coordination_number: int
    mean_coordination_number: float
    mean_hapticity_max: float
    frac_sandwich: float
    frac_carborane: float
    mean_n_P: float
    mean_n_O: float
    mean_n_N: float
    mean_n_B: float


class _ClusterFingerprintReport(BaseModel):
    reducer: str
    min_cluster_size: int
    n_points: int
    n_clusters: int
    n_noise: int
    mean_donor_set_purity: float
    mean_donor_set_entropy_bits: float
    clusters: list[_ClusterFingerprint]


class _GranularityRow(BaseModel):
    min_cluster_size: int
    n_clusters: int
    noise_fraction: float
    mean_donor_set_purity: float
    mean_donor_set_entropy_bits: float
    recommended: bool = False


class _ClusterGranularitySweepReport(BaseModel):
    reducer: str
    n_points: int
    recommended_min_cluster_size: int
    rows: list[_GranularityRow]


class _AxisSummary(BaseModel):
    name: str
    weighted_mean_purity: float
    n_clusters_won: int


class _ClusterAxisRow(BaseModel):
    cluster_id: int
    size: int
    best_axis: str
    best_category: str
    best_purity: float
    purities: dict[str, float]
    modal_categories: dict[str, str]


class _ClusterAxisAnalysisReport(BaseModel):
    reducer: str
    min_cluster_size: int
    n_points: int
    n_clusters: int
    n_noise: int
    axes: list[_AxisSummary]
    clusters: list[_ClusterAxisRow]


class _BenchmarkProtocol(BaseModel):
    """Pinned settings that two scorecards must share to be comparable."""

    version: str
    dataset_path: str
    reducer: str
    cluster_n_components: int
    cluster_n_neighbors: int
    cluster_min_dist: float
    cluster_metric: str
    use_raw_descriptors: bool = False
    random_state: int | None
    canonical_min_cluster_size: int
    chemistry_axes: list[str]
    pure_island_min_size: int
    pure_island_min_purity: float


class _BenchmarkSweepRow(BaseModel):
    min_cluster_size: int
    n_clusters: int
    noise_fraction: float
    ami_per_axis: dict[str, float]


class _DescriptorStructureBenchmarkReport(BaseModel):
    """Single-file benchmark scorecard, one per model run.

    ``aggregate_chemistry_ami`` is the headline number for ranking models —
    average AMI across the chemistry-meaningful axes. Two scorecards are
    only meaningfully comparable when their ``protocol`` matches.
    """

    protocol: _BenchmarkProtocol
    n_points: int
    aggregate_chemistry_ami: float
    canonical: _BenchmarkSweepRow
    n_pure_islands: dict[str, int]
    sweep: list[_BenchmarkSweepRow]


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
    multiplicity: float


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
    multiplicity = np.asarray(dataset.multiplicity[:n_struct])

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
                    "multiplicity": float(multiplicity[idx]),
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
