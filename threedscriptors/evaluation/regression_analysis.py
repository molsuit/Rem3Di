from collections.abc import Iterable

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from threedscriptors.configuration.architecture_config import HeadType
from threedscriptors.data_handling.dataset import (
    RegressionDataset,
    RegressionWithAuxDataset,
)
from threedscriptors.data_handling.sample import Sample, sample_collate_fn
from threedscriptors.model.regression_models import (
    FullyConnectedBlock,
    MultiTaskRegressionModel,
)


def plot_delta_histogram(reference, prediction):
    """
    Plot a histogram of the differences between the predictions and the true values
    """
    assert reference.shape == prediction.shape

    fig = plt.figure()
    plt.hist(reference - prediction, bins=50)
    plt.xlabel("Difference between prediction and true value")
    plt.ylabel("Frequency")
    return fig


def evaluate_regression_model_on_dataset(
    model: MultiTaskRegressionModel,
    dataset: RegressionWithAuxDataset | RegressionDataset,
):
    assert set([tc.task_name for tc in dataset.dataset_config.tasks]).issubset(
        set(model.multitask_heads.task_list)
    )

    batch_size = 128
    dataloader: Iterable[Sample] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=sample_collate_fn,
    )

    device = "cuda"

    regression_predictions = torch.zeros_like(dataset.regression_targets)

    for batch_idx, samples in enumerate(dataloader):
        embeddings = samples.embeddings.to(device)
        padding_mask = samples.padding_mask.to(device)
        auxillary_data = samples.auxillary_data

        regression_predictions[
            batch_idx * batch_size : (batch_idx + 1) * batch_size, :
        ] = model(embeddings, padding_mask, auxillary_data)

    return regression_predictions


def get_regression_head_activations(model: MultiTaskRegressionModel, dataset):
    activations = {}  # keys will be names of layers

    def get_activation(name):
        """Creates a hook function that saves the output of a layer."""

        def hook(model, inp, output):
            activations[name] = output.detach()

        return hook

    hook_handles = []  # These will be used to remove hooks later

    for head_config in model.multitask_heads.regression_head_configs:
        if head_config.head_type is HeadType.RESIDUAL:
            # apply hookbefore the norm
            task_name = head_config.task_name
            head: torch.nn.Module = model.multitask_heads[task_name]
            for i in len():  # This looks at all layers up to the second to last one (i.e not the output layer)
                submodule: FullyConnectedBlock = head.get_submodule(
                    f"fully_connected_{i}"
                )
                submodule.block.activation.register_forward_hook(
                    get_activation(f"{task_name}_block_{i}_activation_output")
                )

        elif head_config.head_type is HeadType.FULLY_CONNECTED:
            # if head is fully connected, apply the hook after the activation
            raise NotImplementedError

    evaluate_regression_model_on_dataset(model, dataset)

    for handle in hook_handles:
        handle.remove()

    return activations
