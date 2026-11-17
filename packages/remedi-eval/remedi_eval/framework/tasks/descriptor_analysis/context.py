from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

from remedi_data.dataset import Dataset
from remedi_eval.projection import ProjectionConfig

if TYPE_CHECKING:
    from ase import Atoms

    from remedi_eval.framework.tasks.descriptor_analysis.tmqm_chemical_features import (
        MetalEnvironmentFeatures,
    )


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

    dataset: Dataset
    descriptors: np.ndarray
    descriptors_raw: np.ndarray
    projection: np.ndarray | None = None
    file_prefix: str = ""
    cache: dict[str, object] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        dataset: Dataset,
        descriptors: np.ndarray,
        normalization: DescriptorNormalizationConfig | None = None,
        projection_config: ProjectionConfig | None = None,
        file_prefix: str = "",
    ) -> DescriptorAnalysisContext:
        raw = np.asarray(descriptors, dtype=np.float64)
        normalizer = normalization or DescriptorNormalizationConfig()
        normalized = normalizer.apply(raw)

        projection = None
        if projection_config is not None:
            projection = projection_config.project(normalized)

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
        molecules = self.dataset.read_structures(
            np.arange(min(self.structure_count, len(self.dataset)))
        ).to_atoms()
        if len(molecules) != self.structure_count:
            raise ValueError(
                f"{self.structure_count} descriptor rows but the dataset holds "
                f"{len(molecules)} structures"
            )
        return molecules

    @cached_property
    def metal_environment_features(self) -> list[MetalEnvironmentFeatures]:
        """Per-structure metal-environment chemistry (one pymatgen pass)."""
        from remedi_eval.framework.tasks.descriptor_analysis.tmqm_chemical_features import (
            compute_metal_environment_features,
        )

        return compute_metal_environment_features(self.molecules)

    def file_name(self, name: str) -> Path:
        if not self.file_prefix or name.startswith(self.file_prefix):
            return Path(name)
        return Path(f"{self.file_prefix}_{name}")
