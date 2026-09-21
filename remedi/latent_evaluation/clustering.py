from abc import ABC, abstractmethod

import numpy as np
import torch
import umap


class ClusteringCalculator(ABC):
    @staticmethod
    @abstractmethod
    def get_dimensionality_reduction(data_matrix: torch.Tensor, k=2):
        pass


class PCACalculator(ClusteringCalculator):
    @staticmethod
    def get_dimensionality_reduction(data_matrix: torch.Tensor, k: int = 2):
        if isinstance(data_matrix, np.ndarray):
            data_matrix = torch.from_numpy(data_matrix)

        data_matrix = data_matrix.detach()
        (_U, _S, V) = torch.pca_lowrank(data_matrix)
        principal_components = torch.matmul(data_matrix, V[:, :k]).numpy()

        return principal_components


class UMAPCalculator(ClusteringCalculator):
    @staticmethod
    def get_dimensionality_reduction(
        data_matrix: torch.Tensor,
        k: int = 2,
        centered: bool = True,
        *,
        metric: str = "cosine",
        n_neighbors: int = 30,
        min_dist: float = 0.0,
        random_state: int | None = 0,
    ):
        """UMAP projection with the project-wide canonical defaults.

        Stock ``umap.UMAP()`` defaults (euclidean, n_neighbors=15, min_dist=0.1,
        random_state=None) produce projections that don't match the tuned
        descriptor visualizations under ``umap_grid_sweep/`` (cosine, nn=30,
        min_dist=0.0). We default to that tuned recipe — and pin
        ``random_state=0`` so successive runs / the chemiscope viewer / the
        analysis-dir PNGs share the same layout — while letting callers
        override for one-off experiments.
        """
        fit = umap.UMAP(
            n_components=k,
            metric=metric,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        data_matrix = data_matrix.detach().cpu().numpy()
        umap_projection = fit.fit_transform(data_matrix)

        if centered:
            umap_projection -= umap_projection.mean(axis=0, keepdims=True)

        return umap_projection
