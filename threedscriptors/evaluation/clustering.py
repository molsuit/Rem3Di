from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import torch
import umap

from threedscriptors.data_handling.data_utils import get_functional_group_label
from threedscriptors.evaluation.regression_analysis import get_colors_for_predictions


class ClusteringCalculator(ABC):
    @staticmethod
    @abstractmethod
    def get_dimensionality_reduction(data_matrix: torch.Tensor, k=2):
        pass


class PCACalculator(ClusteringCalculator):
    @staticmethod
    def get_dimensionality_reduction(data_tensor: torch.Tensor, k=2):
        data_tensor = data_tensor.detach()
        (U, S, V) = torch.pca_lowrank(data_tensor)
        principal_components = torch.matmul(data_tensor, V[:, :k]).numpy()

        return principal_components


class UMAPCalculator(ClusteringCalculator):
    @staticmethod
    def get_dimensionality_reduction(data_matrix: torch.Tensor, k=2, return_fit = False):
        fit = umap.UMAP(n_components=k)
        data_matrix = data_matrix.detach().cpu().numpy()
        umap_projection = fit.fit_transform(data_matrix)
        if return_fit:
            return umap_projection, fit
        
        
        return umap_projection


def plot_reduced_dimension(principle_components):
    # Plot a scatter plot of the principle components. Color each point according to it molecules type in dataset.mol_ids
    pc1 = principle_components[:, 0]
    pc2 = principle_components[:, 1]

    ## Get unique molecule types and assign them colors
    # types = np.array(mol_ids)
    # unique_types = np.unique(types)
    # cmap = plt.get_cmap(
    #    "tab20"
    # )  # up to 10 distinct colors; switch to 'tab20' if you have more
    #
    ## Build a mapping from type → color
    # color_map = {t: cmap(i % cmap.N) for i, t in enumerate(unique_types)}

    # Map each samples type to its color
    # colors = [color_map[t] for t in types]

    # Create the scatter plot
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(pc1, pc2, edgecolor="k", alpha=0.7, s= 1)

    ## Manually build a legend
    # legend_handles = [
    #    Line2D(
    #        [0],
    #        [0],
    #        marker="o",
    #        color="w",
    #        label=moltype,
    #        markerfacecolor=color_map[moltype],
    #        markersize=8,
    #        markeredgecolor="k",
    #    )
    #    for moltype in unique_types
    # ]
    # plt.legend(
    #    handles=legend_handles,
    #    title="Molecule Type",
    #    bbox_to_anchor=(1.05, 1),
    #    loc="upper left",
    # )

    # Labeling
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Scatter Plot of Molecule Descriptors")
    plt.tight_layout()

    return fig


def plot_reduced_dimension_chiral_molecules():
    pass


def plot_reduced_dimension_with_with_regression_labels(principle_components, predictions):

    fig = plt.figure()

    colors = get_colors_for_predictions(predictions)

    plt.scatter(principle_components[:,0], principle_components[:,1], c = colors)

    plt.xlabel("Reduced Dimension 1")
    plt.ylabel("Reduced Dimension 2")


    return fig


def plot_reduced_dimension_functional_group_comparison(reduced_dimensions, smiles):
    functional_group_indices = get_functional_group_label(smiles)
    fig = plt.figure()

    for functional_group_label, mol_indices in functional_group_indices.items():
        plt.scatter(
            x=reduced_dimensions[mol_indices, 0],
            y=reduced_dimensions[mol_indices, 1],
            label=functional_group_label,
        )

    for idx, smile_string in enumerate(smiles):
        plt.annotate(smile_string, xy=reduced_dimensions[idx, 0:2])

    plt.legend()
    plt.tight_layout()
    plt.xlabel("Reduced Dim 1")
    plt.ylabel("Reduced Dim 2")

    return fig
