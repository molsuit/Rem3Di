from collections.abc import Iterable


from threedscriptors.model.remedi_model import REM3DIModel

import numpy as np
import torch
from torch.utils.data import DataLoader

from threedscriptors.data_handling.dataset import (
    AtomicEmbeddingDataset,
    RegressionDataset,
    RegressionWithAuxDataset,
)
from threedscriptors.data_handling.sample import Sample, sample_collate_fn
from threedscriptors.model.model_output import ModelOutput
from threedscriptors.model.regression_models import (
    MultiTaskRegressionModel,
)
from threedscriptors.model.encoder import TransformerEncoder


def evaluate_regression_model_on_dataset(
    model: MultiTaskRegressionModel,
    dataset: RegressionWithAuxDataset | RegressionDataset,
    device="cuda",
    undo_standardization = False
):

    # returns the predictions of the model on dataset in standardized units

    assert set([tc.task_name for tc in dataset.dataset_config.tasks]).issubset(
        set(model.multitask_heads.task_list)
    )
    model.to(device)
    model.eval()

    batch_size = 256
    dataloader: Iterable[Sample] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=sample_collate_fn,
    )

    regression_predictions = torch.zeros_like(dataset.regression_targets)

    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            samples.to_(device)

            if undo_standardization:
                output : ModelOutput = model.inference(samples)
            else:
                output : ModelOutput = model(samples)

            regression_predictions[
                batch_idx * batch_size : (batch_idx + 1) * batch_size, :
            ] = output.regression_predictions

    return regression_predictions


def evaluate_molecular_descriptor_on_dataset(
    model: REM3DIModel, dataset: AtomicEmbeddingDataset, device="cuda"
):
    batch_size = min(64, len(dataset))
    dataloader: Iterable[Sample] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=sample_collate_fn,
    )

    model.to(device)
    model.eval()

    descriptors = torch.zeros(
        size=(len(dataset), model.encoder.aggregator.config.output_dim)
    )

    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            samples.to_(device)

            molecular_descriptor = model(samples)

            descriptors[batch_idx * batch_size : (batch_idx + 1) * batch_size] = molecular_descriptor


    return descriptors



def evaluate_atomic_descriptors(
    model: MultiTaskRegressionModel,
    dataset: RegressionWithAuxDataset | RegressionDataset,
    device="cuda",
):

    model.to(device)
    model.eval()

    batch_size = 256
    dataloader: Iterable[Sample] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=sample_collate_fn,
    )

    regression_predictions = torch.zeros(dataset.dataset_config.N_molecules, dataset.dataset_config.max_atoms, model.preprocessor.config.output_irreps_dim)

    print(regression_predictions.shape)

    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            embeddings = samples.embeddings.to(device)

            batch_size = embeddings.shape[0]

            regression_predictions[
                batch_idx * batch_size : (batch_idx + 1) * batch_size, :
            ] = model.preprocessor(embeddings)

    return regression_predictions





def calculate_fingerprint_uncertainty(
    encoder: TransformerEncoder, dataset: RegressionDataset
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

    mean_descriptor, std_dev_descriptor = compute_class_std(
        data=global_descriptors, class_ids=smiles_class
    )

    return mean_descriptor, std_dev_descriptor


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



def capacity_diagnostics(Z, bins=128, dead_thr=0.2, eps=1e-12):
    """
    Estimate information utilisation of a latent space.

    Parameters
    ----------
    Z : array_like, shape (N_graphs, d)
        Graph-level latent vectors (after pooling).
    bins : int or sequence
        Number of histogram bins per dimension (power of two recommended).
    dead_thr : float
        Fraction of per-dim max entropy below which a dimension is flagged 'dead'.
    eps : float
        Numerical jitter to avoid log(0) / divide-by-zero.

    Returns
    -------
    H_tot : float
        Sum of marginal Shannon entropies (bits).
    utilisation : float
        H_tot divided by effective capacity.
    dead_dims : int
        Count of low-entropy ('dead') coordinates.
    """



    Z = np.asarray(Z, dtype=np.float64)
    Z = Z - np.mean(Z, axis = 0)

    N, d = Z.shape

    # --- marginal entropies -------------------------------------------------
    H_i = np.empty(d)
    for j in range(d):
        counts, _ = np.histogram(Z[:, j], bins=bins)
        p = counts / counts.sum()
        H_i[j] = -np.sum(p * np.log2(p + eps))

    H_tot = H_i.sum()

    # --- capacity proxy -----------------------------------------------------
    max_bits_per_dim = np.log2(bins)                # guaranteed float
    cov = np.cov(Z, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    d_eff = (eigvals.sum()**2) / (np.square(eigvals).sum() + eps)


    C_eff = d_eff * max_bits_per_dim

    utilisation = H_tot / (C_eff + eps)
    dead_dims = int((H_i < dead_thr * max_bits_per_dim).sum())

    eig = np.sort(eigvals)[::-1]           # descending

    return H_tot, utilisation, dead_dims, eig, d_eff



def clip_and_log_transform(y: torch.Tensor) -> torch.Tensor:
    """
    Clip to a detection limit and transform to log10 scale.

    Parameters
    ----------
    y : torch.Tensor
        The tensor to be clipped and transformed.
    """
    # Clip negative values to zero
    y_clipped = torch.clamp(y, min=0.0)
    # Log10 transform with +1 offset for zeros
    return torch.log10(y_clipped + 1.0)