from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Annotated, Any, Literal, cast

import numpy as np
from matplotlib.lines import Line2D
from pydantic import BaseModel, ConfigDict, Field

from remedi.latent_evaluation.context import (
    DescriptorAnalysisContext,
)
from remedi.latent_evaluation.tmqm_clustering_utils import (
    get_atomic_num_colors,
    get_block_colors,
    get_metal_center_type,
    get_tm_colormap,
)


@dataclass
class ColorSpec:
    """Inputs for a Matplotlib scatter plot, plus optional legend handles."""

    values: Any  # color array, scalar value array, or single color string
    scatter_kwargs: dict[str, Any]
    legend_handles: list[Line2D] | None = None
    colorbar: bool = False
    colorbar_label: str | None = None


def _metal_center_atomic_numbers(ctx: DescriptorAnalysisContext) -> list[int]:
    """The single transition metal of every structure (cached on the context)."""
    atomic_numbers = ctx.cache.get("metal_center_atomic_numbers")
    if atomic_numbers is None:
        atomic_numbers = get_metal_center_type(ctx.molecules)
        ctx.cache["metal_center_atomic_numbers"] = atomic_numbers
    return cast("list[int]", atomic_numbers)


class _BaseColorProvider(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    label: str | None = None
    """Optional override for the plot title suffix; defaults to a per-provider value."""

    @abstractmethod
    def compute(self, ctx: DescriptorAnalysisContext) -> ColorSpec: ...

    @abstractmethod
    def default_label(self) -> str: ...

    def title_suffix(self) -> str:
        return self.label or self.default_label()


class NumAtomsColor(_BaseColorProvider):
    kind: Literal["num_atoms"] = "num_atoms"

    def default_label(self) -> str:
        return "Number of atoms"

    def compute(self, ctx: DescriptorAnalysisContext) -> ColorSpec:
        sizes = np.asarray([len(m) for m in ctx.molecules], dtype=np.int64)
        return ColorSpec(
            values=sizes,
            scatter_kwargs={"cmap": "viridis"},
            colorbar=True,
            colorbar_label=self.title_suffix(),
        )


class MetalCenterElementColor(_BaseColorProvider):
    kind: Literal["metal_center_element"] = "metal_center_element"

    def default_label(self) -> str:
        return "Metal center (element)"

    def compute(self, ctx: DescriptorAnalysisContext) -> ColorSpec:
        colors, handles = get_atomic_num_colors(_metal_center_atomic_numbers(ctx))
        return ColorSpec(
            values=colors,
            scatter_kwargs={},
            legend_handles=handles,
        )


class DBlockColor(_BaseColorProvider):
    kind: Literal["d_block"] = "d_block"

    def default_label(self) -> str:
        return "Metal center (d-block)"

    def compute(self, ctx: DescriptorAnalysisContext) -> ColorSpec:
        colors = get_block_colors(_metal_center_atomic_numbers(ctx))
        return ColorSpec(
            values=colors,
            scatter_kwargs={"cmap": "tab10"},
        )


class MetalCenterAtomicNumberColor(_BaseColorProvider):
    """Colors by atomic number using the custom d-row TM colormap."""

    kind: Literal["metal_center_atomic_number"] = "metal_center_atomic_number"

    def default_label(self) -> str:
        return "Metal center (atomic number)"

    def compute(self, ctx: DescriptorAnalysisContext) -> ColorSpec:
        cmap, norm = get_tm_colormap()
        return ColorSpec(
            values=np.asarray(_metal_center_atomic_numbers(ctx), dtype=np.int64),
            scatter_kwargs={"cmap": cmap, "norm": norm},
            colorbar=True,
            colorbar_label="Atomic number",
        )


class CoordinationNumberColor(_BaseColorProvider):
    """First-sphere coordination number of the metal (-1 where unresolved)."""

    kind: Literal["coordination_number"] = "coordination_number"

    def default_label(self) -> str:
        return "Coordination number"

    def compute(self, ctx: DescriptorAnalysisContext) -> ColorSpec:
        coordination_numbers = [
            features.coordination_number for features in ctx.metal_environment_features
        ]
        return ColorSpec(
            values=np.asarray(coordination_numbers, dtype=np.int64),
            scatter_kwargs={"cmap": "tab10"},
            colorbar=True,
            colorbar_label=self.title_suffix(),
        )


class RegressionTargetColor(_BaseColorProvider):
    kind: Literal["regression_target"] = "regression_target"
    target_index: int = 0
    target_name: str | None = None
    cmap: str = "viridis"

    def default_label(self) -> str:
        if self.target_name is not None:
            return self.target_name
        return f"Regression target #{self.target_index}"

    def compute(self, ctx: DescriptorAnalysisContext) -> ColorSpec:
        targets = ctx.dataset.targets_system
        if targets is None:
            raise ValueError(
                "Dataset has no system-level regression targets but a "
                "RegressionTargetColor provider was requested."
            )

        values = np.asarray(
            targets[: ctx.structure_count, self.target_index], dtype=np.float64
        )
        return ColorSpec(
            values=values,
            scatter_kwargs={"cmap": self.cmap},
            colorbar=True,
            colorbar_label=self.title_suffix(),
        )


ColorProvider = Annotated[
    NumAtomsColor
    | MetalCenterElementColor
    | DBlockColor
    | MetalCenterAtomicNumberColor
    | CoordinationNumberColor
    | RegressionTargetColor,
    Field(discriminator="kind"),
]
