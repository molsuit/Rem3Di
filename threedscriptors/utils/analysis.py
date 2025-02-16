import matplotlib.pyplot as plt
import numpy as np
import torch

from threedscriptors.data_handling.dataset import RegressionAtomEmbeddingDataset
from threedscriptors.model.model import TransformerEncoder


def get_PCA(data_mat, k):
    (U, S, V) = torch.pca_lowrank(data_mat)
    k = 2
    principle_components = torch.matmul(data_mat, V[:, :k]).numpy()

    return principle_components


def get_fingerprint_correlation():
    """
    Might be interesting to see the correlation between our global descriptors and the morgan or ecfp fingerprints
    """

    pass


def plot_delta_histogram(delta):
    """
    Plot a histogram of the differences between the predictions and the true values
    """
    fig = plt.figure()
    plt.hist(delta, bins=50)
    plt.xlabel("Difference between prediction and true value")
    plt.ylabel("Frequency")
    return fig


def calculate_fingerprint_uncertainty(
    encoder: TransformerEncoder, dataset: RegressionAtomEmbeddingDataset
):
    # for all smiles in the smiles list, get the corresponding unique dataset id

    smiles_list = dataset.smiles_list
    smiles_hash = {smiles: idx for idx, smiles in enumerate(set(smiles_list))}
    smiles_class = np.array([smiles_hash[smiles] for smiles in smiles_list])

    global_descriptors = np.zeros(
        shape=(len(smiles_list), encoder.architecture_config.embedding_size)
    )

    encoder.eval()
    with torch.no_grad():
        for idx, embedding, padding_mask in enumerate(
            zip(dataset.embeddings, dataset.padding_mask, strict=False)
        ):
            global_descriptors[idx, :] = encoder.forward(embedding, padding_mask)

    mean_discriptor, std_dev_descriptor = compute_class_std(
        data=global_descriptors, class_ids=smiles_class
    )

    return mean_discriptor, std_dev_descriptor


def compute_class_std(data, class_ids):
    data = np.asarray(data)
    class_ids = np.asarray(class_ids).reshape(
        -1,
    )

    classes = np.unique(class_ids)

    class_std_dev = np.zeros(shape=(len(classes), data.shape[1]))
    class_mean = np.zeros(shape=(len(classes), data.shape[1]))

    for idx, class_id in enumerate(classes):
        mask = np.where(class_ids == class_id, True, False)
        class_std_dev[idx, :] = np.std(data[mask, :], axis=0)
        class_mean[idx, :] = np.mean(data[mask, :], axis=0)

    return class_mean, class_std_dev
