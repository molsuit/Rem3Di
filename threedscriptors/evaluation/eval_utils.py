from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader
from threedscriptors.data_handling.dataset import (
    AtomicEmbeddingDataset,
    RegressionDataset,
    RegressionWithAuxDataset,
)
from threedscriptors.data_handling.sample import Sample, sample_collate_fn
from threedscriptors.model.regression_models import (
    MultiTaskRegressionModel,
)


def evaluate_regression_model_on_dataset(
    model: MultiTaskRegressionModel,
    dataset: RegressionWithAuxDataset | RegressionDataset,
    device="cuda",
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
            embeddings = samples.embeddings.to(device)
            padding_mask = samples.padding_mask.to(device)
            auxillary_data = samples.auxillary_data
    
            regression_predictions[
                batch_idx * batch_size : (batch_idx + 1) * batch_size, :
            ] = model(embeddings, padding_mask, auxillary_data)

    return regression_predictions


def evaluate_molecular_descriptor_on_dataset(
    model: MultiTaskRegressionModel, dataset: AtomicEmbeddingDataset, device="cuda"
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
        size=(len(dataset), model.global_aggregator.config.output_dim)
    )

    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            embeddings = samples.embeddings.to(device)
            padding_mask = samples.padding_mask.to(device)

            descriptors[batch_idx * batch_size : (batch_idx + 1) * batch_size] = (
                model.get_molecular_descriptor(embeddings, padding_mask).detach().cpu()
            )

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
 
            regression_predictions[
                batch_idx * batch_size : (batch_idx + 1) * batch_size, :
            ] = model.preprocessor(embeddings)

    return regression_predictions
