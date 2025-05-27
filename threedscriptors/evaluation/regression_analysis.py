from collections.abc import Callable
from collections import defaultdict

import matplotlib.pyplot as plt
import torch

from threedscriptors.configuration.architecture_config import HeadType
from threedscriptors.model.regression_models import (
    FullyConnectedBlock,
    MultiTaskRegressionModel,
    ResidualBlock,
)

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

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


def add_regression_head_activations_hooks(
    model: MultiTaskRegressionModel,
) -> tuple[dict, Callable]:
    activations : dict[str, list[torch.Tensor]] = defaultdict(list) # keys will be names of layers

    def get_activation(name):
        """Creates a hook function that saves the output of a layer."""

        def hook(model, inp, output):
            activations[name].append(output.detach().cpu())

        return hook

    for head_config in model.multitask_heads.regression_head_configs:
        if head_config.head_type is HeadType.FULLY_CONNECTED:
            # if head is fully connected, apply the hook after the activation
            task_name = head_config.task_name
            head: torch.nn.Module = model.multitask_heads.task_heads[task_name]

            for i in range(
                len(head_config.hidden_dimensions)
            ):  # This looks at all layers up to the second to last one (i.e not the output layer)
                submodule: FullyConnectedBlock = head.get_submodule(
                    f"fully_connected_{i}"
                )
                submodule.block.activation.register_forward_hook(
                    get_activation(f"{task_name}_block_{i}_activation_output")
                )

        elif head_config.head_type is HeadType.RESIDUAL:
            # apply hookbefore the norm
            # if head is fully connected, apply the hook after the activation
            task_name = head_config.task_name
            head: torch.nn.Module = model.multitask_heads.task_heads[task_name]

            for i in range(
                len(head_config.hidden_dimensions)
            ):  # This looks at all layers up to the second to last one (i.e not the output layer)
                submodule: ResidualBlock = head.get_submodule(f"residual_{i}")
                submodule.norm.register_forward_hook(
                    get_activation(f"{task_name}_block_{i}_activation_output")
                )

    return activations





def get_colors_for_predictions(predictions):

    norm = Normalize(vmin=predictions.min(), vmax=predictions.max())
    # 2. Pick a colormap (you can swap 'viridis' for any Matplotlib cmap)
    cmap = get_cmap('viridis')
    # 3. Turn it into a ScalarMappable, so you can map values → RGBA
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    # 4. Get colors for each prediction
    colors = mappable.to_rgba(predictions)

    return colors