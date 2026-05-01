from __future__ import annotations

from abc import abstractmethod
from typing import Annotated, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

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
    file_name: str = "capacity_diagnostic.yaml"
    plot_per_dim: bool = True
    plot_explained_variance: bool = True

    def run(self, ctx: DescriptorAnalysisContext) -> list[EvalResult]:
        report = run_latent_space_capacity_diagnostic(
            ctx.descriptors_raw,
            bins=self.bins,
            dead_thr=self.dead_threshold,
            standardize=self.standardize,
        )

        results: list[EvalResult] = [
            PydanticResult(file_name=ctx.file_name(self.file_name), obj=report),
        ]

        if self.plot_per_dim:
            results.append(
                FigureResult(
                    file_name=ctx.file_name("capacity_per_dim.png"),
                    figure=_plot_per_dim_entropy(report),
                )
            )
        if self.plot_explained_variance:
            results.append(
                FigureResult(
                    file_name=ctx.file_name("capacity_explained_variance.png"),
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


DescriptorAnalysisTask = Annotated[
    CapacityDiagnosticTask
    | DescriptorDistributionTask
    | ProjectionPlotTask
    | ChemiscopeProjectionTask,
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


def _plot_per_dim_entropy(report: LatentCapacityReportModel) -> Any:
    fig, ax = plt.subplots(figsize=(10, 4))
    order = np.argsort(report.H_norm_per_dim)[::-1]
    ax.bar(np.arange(len(order)), report.H_norm_per_dim[order], width=1.0)
    ax.axhline(0.2, color="red", linestyle="--", linewidth=1, label="dead threshold")
    ax.set_xlabel("Latent dimension (sorted)")
    ax.set_ylabel("H_i / log2(bins_i)")
    ax.set_title(
        f"Per-dim entropy utilisation "
        f"(d_eff={report.d_eff:.1f}, dead={report.dead_dims}, util={report.utilisation:.2f})"
    )
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
