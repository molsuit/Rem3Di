import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine, euclidean, jaccard


def tanimoto_similarity(a: np.ndarray, b: np.ndarray) -> float:
    print(a.shape)
    print(b.shape)
    return 1 - jaccard(a, b)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return (2 - cosine(a, b)) / 2


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 1 / (1 + euclidean(a, b))


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Given X: (N, D) array of descriptors,
    returns sim: (N, N) where
      sim[i,j] =  <X[i], X[j]> / (||X[i]|| * ||X[j]||)
    and then linearly shifted to [0,1].
    """
    # 1) dot products: (N, D) @ (D, N) --> (N, N)
    dotprods = X @ X.T

    # 2) norms: shape (N,)
    norms = np.linalg.norm(X, axis=1)

    # 3) outer of norms: shape (N,N)
    norm_matrix = np.outer(norms, norms)

    # 4) elementwise division
    cosine = dotprods / norm_matrix

    # 5) linear shift into [0,1]
    return (cosine + 1.0) / 2.0


def plot_similarity_matrix(sim_matrix, cmap="inferno"):
    """
    Plot a heatmap of the pairwise similarity matrix, ensuring full display.

    Parameters:
    - sim_matrix: square numpy array of shape (N, N)
    - labels: optional list of length N for tick labels
    - cell_size: size (inches) per row/column
    - max_fig_size: maximum figure dimension (inches)
    - cmap: Matplotlib colormap name string
    """
    fig, ax = plt.subplots()

    im = ax.imshow(
        sim_matrix, origin="upper", aspect="equal", interpolation="nearest", cmap=cmap
    )
    fig.colorbar(im, ax=ax)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel("Molecule index")
    ax.set_ylabel("Molecule index")
    ax.set_title("Pairwise Similarity Matrix")
    plt.tight_layout()

    return fig
