from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID
from threedscriptors.data_handling.sample import (
    Sample,
    paired_sample_collate_fn,
    sample_collate_fn,
    yield_molecules_collate_fn,
)
from threedscriptors.model.model_output import ModelOutput
from threedscriptors.model.molecule_difference_regressor import (
    MolecularDifferenceRegressor,
)
from threedscriptors.model.pair_encoder import TransformerPairEncoder
from threedscriptors.model.regression_models import (
    MultiTaskRegressionModel,
)
from threedscriptors.model.remedi_model import REM3DIModel


def evaluate_regression_model_on_dataset(
    model: MultiTaskRegressionModel,
    dataset: TrainingMoleculeDataset,
    device="cuda",
    undo_standardization=False,
):
    # returns the predictions of the model on dataset in standardized units

    assert set([tc.task_name for tc in dataset.dataset_config.tasks]).issubset(
        set(model.multitask_heads.task_list)
    )
    model.to(device)
    model.eval()

    batch_size = 256

    collate_fn = sample_collate_fn
    dataloader: Iterable[Sample] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    regression_predictions = torch.zeros_like(dataset.regression_targets)

    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            samples.to_(device)

            if undo_standardization:
                output: ModelOutput = model.inference(samples)
            else:
                output: ModelOutput = model(samples)

            regression_predictions[
                batch_idx * batch_size : (batch_idx + 1) * batch_size, :
            ] = output.regression_predictions

    return regression_predictions


def evaluate_molecular_descriptor_on_dataset(
    model: REM3DIModel, dataset: TrainingMoleculeDataset, device="cuda"
):
    """Run the encoder over `dataset` and return descriptors as a flat
    `(N, L * d_out)` tensor — `L` seed tokens are concatenated per molecule."""
    batch_size = min(64, len(dataset))

    dataloader: Iterable[Sample] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=yield_molecules_collate_fn,
    )

    model.to(device)
    model.eval()

    aggregator = model.encoder.aggregator
    flat_dim = aggregator.seq_len * aggregator.d_out
    descriptors = torch.zeros(size=(len(dataset), flat_dim))

    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            samples.to_(device)

            model_output = model(samples)
            descriptors[batch_idx * batch_size : (batch_idx + 1) * batch_size] = (
                model_output.molecular_descriptor.flat
            )

    return descriptors


def evaluate_molecule_difference_on_dataset(
    model: REM3DIModel,
    dataset,
    molecular_difference_regressor: MolecularDifferenceRegressor,
    device="cuda",
):
    batch_size = min(64, len(dataset))

    collate_fn = paired_sample_collate_fn

    dataloader: Iterable[Sample] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    model.to(device)
    model.eval()
    molecular_difference_regressor.eval()

    differences = torch.zeros(size=(len(dataset), 1))
    print(differences.shape)

    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            samples.to_(device)

            descriptors = model(samples).molecular_descriptor

            differences[batch_idx * batch_size : (batch_idx + 1) * batch_size] = (
                molecular_difference_regressor(
                    descriptors,
                    samples.auxillary_data["cmrt"],
                )
            )

    return differences


def evaluate_atomic_descriptors(
    model: MultiTaskRegressionModel,
    dataset: TrainingMoleculeDataset,
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

    regression_predictions = torch.zeros(
        dataset.dataset_config.N_molecules,
        dataset.dataset_config.max_atoms,
        model.preprocessor.config.output_irreps_dim,
    )

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
    encoder: TransformerPairEncoder, dataset: TrainingMoleculeDataset
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


def average_over_conformers(
    structure_ids: list[StructureID], predictions: torch.Tensor
):
    classes = [(sid.molecule_id, sid.enantiomer_id) for sid in structure_ids]
    class_ids = {mol_e_id: i for i, mol_e_id in enumerate(set(classes))}

    class_id_per_mol = []

    for sid in structure_ids:
        class_id_per_mol.append(class_ids[(sid.molecule_id, sid.enantiomer_id)])

    class_id_per_mol = torch.as_tensor(class_id_per_mol).reshape(
        -1,
    )

    print(class_id_per_mol)
    for class_id in class_ids.values():
        mask = torch.where(class_id_per_mol == class_id)
        class_mean = torch.mean(predictions[mask], dim=0)
        print(class_mean)

        predictions[mask] = class_mean

    return predictions


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
