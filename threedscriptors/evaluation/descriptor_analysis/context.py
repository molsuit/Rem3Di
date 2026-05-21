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
    """Configures the 2D projection cached on the context.

    UMAP defaults match the settled descriptor-visualization recipe
    (``cosine`` / ``n_neighbors=30`` / ``min_dist=0``, seed 0) so the chemiscope
    viewer, the analysis-dir PNGs, and any cross-model comparison share one
    deterministic layout. Override per-task for one-off experiments.
    """

    method: Literal["umap", "pca"] = "umap"
    n_components: int = 2
    center: bool = True
    metric: str = "cosine"
    n_neighbors: int = 30
    min_dist: float = 0.0
    random_state: int | None = Field(default=0)

    def compute(self, descriptors: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(descriptors))
        if self.method == "pca":
            projection = PCACalculator.get_dimensionality_reduction(
                tensor, k=self.n_components
            )
        else:
            projection = UMAPCalculator.get_dimensionality_reduction(
                tensor,
                k=self.n_components,
                centered=self.center,
                metric=self.metric,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                random_state=self.random_state,
            )
        return np.asarray(projection)


@dataclass
class DescriptorAnalysisContext:
    """Bundles the inputs every descriptor analysis task needs.

    ``descriptors`` is the normalized matrix used for projection / metrics.
    ``descriptors_raw`` is kept around so capacity diagnostics see the
    untouched representation. ``sample_indices`` maps each descriptor row to
    its index in ``dataset`` when descriptors were computed on a subset of the
    dataset (None ⇒ identity, descriptors cover the full dataset in order).
    """

    dataset: MoleculeDataset
    descriptors: np.ndarray
    descriptors_raw: np.ndarray
    projection: np.ndarray | None = None
    file_prefix: str = ""
    sample_indices: np.ndarray | None = None
    cache: dict[str, object] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        dataset: MoleculeDataset,
        descriptors: np.ndarray | torch.Tensor,
        normalization: DescriptorNormalizationConfig | None = None,
        projection_config: ProjectionConfig | None = None,
        file_prefix: str = "",
        sample_indices: np.ndarray | None = None,
    ) -> DescriptorAnalysisContext:
        raw = _to_numpy(descriptors).astype(np.float64, copy=False)
        normalizer = normalization or DescriptorNormalizationConfig()
        normalized = normalizer.apply(raw)

        indices = None
        if sample_indices is not None:
            indices = np.ascontiguousarray(sample_indices, dtype=np.int64)
            if indices.shape != (normalized.shape[0],):
                raise ValueError(
                    f"sample_indices shape {indices.shape} does not match "
                    f"({normalized.shape[0]},) descriptor rows"
                )

        projection = None
        if projection_config is not None:
            projection = projection_config.compute(normalized)

        return cls(
            dataset=dataset,
            descriptors=normalized,
            descriptors_raw=raw,
            projection=projection,
            file_prefix=file_prefix,
            sample_indices=indices,
        )

    def to_dataset_indices(self, subset_idx: np.ndarray) -> np.ndarray:
        """Map descriptor-row indices to their indices in ``dataset``."""
        subset_idx = np.asarray(subset_idx, dtype=np.int64)
        if self.sample_indices is None:
            return subset_idx
        return self.sample_indices[subset_idx]

    def file_name(self, name: str) -> Path:
        if not self.file_prefix or name.startswith(self.file_prefix):
            return Path(name)
        return Path(f"{self.file_prefix}_{name}")
