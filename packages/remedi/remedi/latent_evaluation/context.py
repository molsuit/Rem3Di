from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from remedi_data.dataset.molecule_dataset import MoleculeDataset

if TYPE_CHECKING:
    from ase import Atoms

    from remedi.latent_evaluation.tmqm_chemical_features import (
        MetalEnvironmentFeatures,
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

    model_config = ConfigDict(extra="forbid")

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
    deterministic layout. ``metric``, ``n_neighbors`` and ``min_dist`` only
    apply to UMAP. ``center`` shifts the projection to zero mean (for PCA:
    ``False`` projects the uncentred descriptors onto the principal axes).
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["umap", "pca"] = "umap"
    n_components: int = Field(default=2, gt=0)
    center: bool = True
    metric: str = "cosine"
    n_neighbors: int = 30
    min_dist: float = 0.0
    random_state: int | None = Field(default=0)

    def compute(self, descriptors: np.ndarray) -> np.ndarray:
        """Project ``descriptors`` (rows are samples) to ``n_components`` columns.

        Raises:
            ValueError: if ``n_components`` exceeds what the data supports.
        """
        if self.method == "pca":
            from sklearn.decomposition import PCA

            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            pca.fit(descriptors)
            offset = pca.mean_ if self.center else 0.0
            return (descriptors - offset) @ pca.components_.T

        import umap

        projection = np.asarray(
            umap.UMAP(
                n_components=self.n_components,
                metric=self.metric,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                random_state=self.random_state,
            ).fit_transform(descriptors)
        )
        if self.center:
            projection = projection - projection.mean(axis=0, keepdims=True)
        return projection


@dataclass
class DescriptorAnalysisContext:
    """Bundles the inputs every descriptor analysis task needs.

    ``descriptors`` is the normalized matrix used for projection / metrics.
    ``descriptors_raw`` is kept around so capacity diagnostics see the
    untouched representation. Row ``i`` of both is structure ``i`` of
    ``dataset``: the descriptors cover the first ``structure_count`` structures.
    Per-structure inputs (molecules, metal environments) are read once, for
    exactly those structures, and shared by every task.
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

    @property
    def structure_count(self) -> int:
        """Number of descriptor rows, the first structures of ``dataset``."""
        return int(self.descriptors.shape[0])

    @cached_property
    def molecules(self) -> list[Atoms]:
        """The structures behind the descriptor rows, in row order (read once).

        Raises:
            ValueError: if the dataset holds fewer structures than descriptor rows.
        """
        molecules = self.dataset.get_all_molecules(self.structure_count)
        if len(molecules) != self.structure_count:
            raise ValueError(
                f"{self.structure_count} descriptor rows but the dataset holds "
                f"{len(molecules)} structures"
            )
        return molecules

    @cached_property
    def metal_environment_features(self) -> list[MetalEnvironmentFeatures]:
        """Per-structure metal-environment chemistry (one pymatgen pass)."""
        from remedi.latent_evaluation.tmqm_chemical_features import (
            compute_metal_environment_features,
        )

        return compute_metal_environment_features(self.molecules)

    def file_name(self, name: str) -> Path:
        if not self.file_prefix or name.startswith(self.file_prefix):
            return Path(name)
        return Path(f"{self.file_prefix}_{name}")
