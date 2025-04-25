import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from threedscriptors.data_handling.dataset import BaseDataset


def get_PCA(data_mat, k=2):
    (U, S, V) = torch.pca_lowrank(data_mat)
    principle_components = torch.matmul(data_mat, V[:, :k]).numpy()

    return principle_components


def plot_PCAs(principle_components, dataset: BaseDataset):
    # Plot a scatter plot of the principle components. Color each point according to it molecules type in dataset.mol_ids
    pc1 = principle_components[:, 0]
    pc2 = principle_components[:, 1]

    # Get unique molecule types and assign them colors
    types = np.array(dataset.mol_ids[:])
    print(dataset.mol_ids)
    unique_types = np.unique(types)
    print(unique_types.shape)
    cmap = plt.get_cmap(
        "tab20"
    )  # up to 10 distinct colors; switch to 'tab20' if you have more

    # Build a mapping from type → color
    color_map = {t: cmap(i % cmap.N) for i, t in enumerate(unique_types)}

    # Map each samples type to its color
    colors = [color_map[t] for t in types]

    # Create the scatter plot
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(pc1, pc2, c=colors, s=50, edgecolor="k", alpha=0.7)

    # Manually build a legend
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=moltype,
            markerfacecolor=color_map[moltype],
            markersize=8,
            markeredgecolor="k",
        )
        for moltype in unique_types
    ]
    plt.legend(
        handles=legend_handles,
        title="Molecule Type",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    # Labeling
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Scatter Plot of Molecule Descriptors")
    plt.tight_layout()

    return fig
