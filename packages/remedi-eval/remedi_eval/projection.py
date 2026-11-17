"""Low-dimensional projections of a descriptor matrix: one config union, one ``project``.

``ProjectionConfig`` (``kind: umap`` or ``kind: pca``) is what every analysis
that projects descriptors takes: the 2D layout of the descriptor analyses, the
higher-dimensional embedding their clustering runs on, and the UMAP of the
BitBIRCH fingerprints in the dataset analyses. Each use sets its own defaults;
the :class:`UmapConfig` defaults are the descriptor-visualisation recipe
(``cosine``, ``n_neighbors=30``, ``min_dist=0``, seed 0), so the chemiscope
viewer, the analysis PNGs and cross-model comparisons share one deterministic
layout.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class UmapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["umap"] = "umap"
    n_components: int = Field(default=2, gt=0)
    n_neighbors: int = Field(default=30, ge=2)
    min_dist: float = Field(default=0.0, ge=0.0, le=1.0)
    metric: str = "cosine"
    random_state: int | None = 0

    def project(self, matrix: np.ndarray) -> np.ndarray:
        """``(n_rows, n_components)`` UMAP coordinates of ``matrix``'s rows."""
        import umap

        return np.asarray(
            umap.UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=self.random_state,
            ).fit_transform(matrix)
        )

    def label(self) -> str:
        return f"{self.n_components}D {self.metric} UMAP"


class PcaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["pca"] = "pca"
    n_components: int = Field(default=2, gt=0)
    random_state: int | None = 0

    def project(self, matrix: np.ndarray) -> np.ndarray:
        """The centred rows of ``matrix`` on the first ``n_components`` principal axes.

        Raises:
            ValueError: if ``n_components`` exceeds what the data supports.
        """
        from sklearn.decomposition import PCA

        return np.asarray(
            PCA(
                n_components=self.n_components, random_state=self.random_state
            ).fit_transform(matrix)
        )

    def label(self) -> str:
        return f"{self.n_components}D PCA"


ProjectionConfig = Annotated[UmapConfig | PcaConfig, Field(discriminator="kind")]
