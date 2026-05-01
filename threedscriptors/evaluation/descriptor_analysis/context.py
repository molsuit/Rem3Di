from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, Field

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.evaluation.descriptor_analysis.clustering import (
    PCACalculator,
    UMAPCalculator,
)


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _z_score(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return (x - mean) / (std + eps)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


class DescriptorNormalizationConfig(BaseModel):
    """Pre-projection normalization applied to descriptors."""

    z_score: bool = True
    l2_normalize: bool = True

    def apply(self, descriptors: np.ndarray) -> np.ndarray:
        out = descriptors
        if self.z_score:
            out = _z_score(out)
        if self.l2_normalize:
            out = _l2_normalize(out)
        return out


class ProjectionConfig(BaseModel):
    """Configures the 2D projection cached on the context."""

    method: Literal["umap", "pca"] = "umap"
    n_components: int = 2
    center: bool = True
    random_state: int | None = Field(default=None)

    def compute(self, descriptors: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(descriptors))
        if self.method == "pca":
            projection = PCACalculator.get_dimensionality_reduction(
                tensor, k=self.n_components
            )
        else:
            calculator = UMAPCalculator()
            projection = calculator.get_dimensionality_reduction(
                tensor, k=self.n_components, centered=self.center
            )
        return np.asarray(projection)


@dataclass
class DescriptorAnalysisContext:
    """Bundles the inputs every descriptor analysis task needs.

    ``descriptors`` is the normalized matrix used for projection / metrics.
    ``descriptors_raw`` is kept around so capacity diagnostics see the
    untouched representation.
    """

    dataset: MoleculeDataset
    descriptors: np.ndarray
    descriptors_raw: np.ndarray
    projection: np.ndarray | None = None
    file_prefix: str = ""
    cache: dict[str, object] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        dataset: MoleculeDataset,
        descriptors: np.ndarray | torch.Tensor,
        normalization: DescriptorNormalizationConfig | None = None,
        projection_config: ProjectionConfig | None = None,
        file_prefix: str = "",
    ) -> DescriptorAnalysisContext:
        raw = _to_numpy(descriptors).astype(np.float64, copy=False)
        normalizer = normalization or DescriptorNormalizationConfig()
        normalized = normalizer.apply(raw)

        projection = None
        if projection_config is not None:
            projection = projection_config.compute(normalized)

        return cls(
            dataset=dataset,
            descriptors=normalized,
            descriptors_raw=raw,
            projection=projection,
            file_prefix=file_prefix,
        )

    def file_name(self, name: str) -> Path:
        if not self.file_prefix or name.startswith(self.file_prefix):
            return Path(name)
        return Path(f"{self.file_prefix}_{name}")
