import matplotlib.pyplot as plt

from remedi.data_handling.data_utils import get_functional_group_label


def plot_reduced_dimension_3d(principle_components, color="k", suptitle=None, **kwargs):
    """
    Parameters
    ----------
    principle_components : (N, 3) array‑like
        Coordinates for the first three UMAP/PC components.
    color : str | sequence, optional
        • Single Matplotlib colour → whole cloud uses that colour
        • Sequence of scalars or colour specs → each point coloured individually.
    suptitle : str, optional
        Figure‑level title.
    **kwargs
        Extra keyword arguments forwarded to `ax.scatter`
        (e.g. `s`, `cmap`, `alpha`, `marker`, …).

    Returns
    -------
    matplotlib.figure.Figure
        Handle to the created figure.
    """
    if principle_components.shape[1] < 3:
        raise ValueError("Need at least three components for a 3‑D plot.")

    pc1, pc2, pc3 = (
        principle_components[:, 0],
        principle_components[:, 1],
        principle_components[:, 2],
    )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 3‑D scatter
    ax.scatter(pc1, pc2, pc3, c=color, alpha=0.7, s=0.4, **kwargs)

    # Labelling
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_zlabel("UMAP3")
    ax.set_title("3-D UMAP Plot of Molecule Descriptors")
    fig.tight_layout()

    if suptitle is not None:
        fig.suptitle(suptitle)

    return fig


def plot_reduced_dimension(
    principle_components, color="k", suptitle=None, handles=None, **kwargs
):
    # Plot a scatter plot of the principle components. Color each point according to it molecules type in dataset.mol_ids
    pc1 = principle_components[:, 0]
    pc2 = principle_components[:, 1]

    # Create the scatter plot
    fig = plt.figure(figsize=(8, 6))

    plt.scatter(pc1, pc2, c=color, alpha=0.7, s=0.4, **kwargs)

    # Labeling
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP Plot of Molecule Descriptors")

    if handles is not None:
        plt.legend(handles=handles)

    if suptitle is not None:
        plt.title(f"UMAP Plot of Molecule Descriptors {suptitle}")

    plt.tight_layout()
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


#  coloring by

# molecular_weight
# regression_labels
# regression predictions
