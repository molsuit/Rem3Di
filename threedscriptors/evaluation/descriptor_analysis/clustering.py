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
    def get_dimensionality_reduction(data_tensor: torch.Tensor, k=2):
        if isinstance(data_tensor, np.ndarray):
            data_tensor = torch.from_numpy(data_tensor)

        data_tensor = data_tensor.detach()
        (U, S, V) = torch.pca_lowrank(data_tensor)
        principal_components = torch.matmul(data_tensor, V[:, :k]).numpy()

        return principal_components


class UMAPCalculator(ClusteringCalculator):
    @staticmethod
    def get_dimensionality_reduction(
        data_matrix: torch.Tensor, k=2, centered: bool = True
    ):
        fit = umap.UMAP(n_components=k)
        data_matrix = data_matrix.detach().cpu().numpy()
        umap_projection = fit.fit_transform(data_matrix)

        if centered:
            umap_projection -= umap_projection.mean(axis=0, keepdims=True)

        return umap_projection
