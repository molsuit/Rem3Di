from collections import OrderedDict
from itertools import pairwise

import torch
import torch.nn as nn

from threedscriptors.configuration.architecture_config import (
    HeadType,
    RegressionHeadConfig,
)
from threedscriptors.model.atomic_descriptor_preprocess import (
    AtomicDescriptorPreprocess,
)
from threedscriptors.model.global_aggregator import GlobalAggregator
from threedscriptors.model.transformer_components import TransformerEncoder


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation_fn: nn.Module):
        super().__init__()
        # Layer normalization applied to the input.
        self.norm = nn.LayerNorm(out_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation_fn

        if in_dim != out_dim:
            self.projection = nn.Linear(in_dim, out_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: normalize the input.
        # Main branch transformation on normalized input.
        out = self.linear(x)
        out = self.activation(out)
        # Residual shortcut
        residual = self.projection(x)
        out = out + residual
        # This activation
        out = self.norm(out)
        # Elementwise addition.
        return out


class FullyConnectedBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation_fn: nn.Module):
        super().__init__()

        self.block = nn.Sequential(
            OrderedDict(
                [
                    ("linear_layer", nn.Linear(in_dim, out_dim)),
                    ("layer_norm", nn.LayerNorm(out_dim)),
                    ("activation", activation_fn),
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        out = self.block(x)
        return out


class MultitaskHeads(nn.Module):
    def __init__(self, regression_head_configs: list[RegressionHeadConfig]):
        super().__init__()
        self.regression_head_configs = regression_head_configs

        self.task_list = [conf.task_name for conf in regression_head_configs]
        self.N_tasks = len(self.task_list)

        self.task_heads = nn.ModuleDict()
        for head_config in regression_head_configs:
            self.task_heads[head_config.task_name] = self.build_head(head_config)

    @staticmethod
    def build_head(
        head_config: RegressionHeadConfig,
    ) -> nn.Module:
        """
        Build a regression head based on the provided configuration.

        Args:
            head_config (RegressionHeadConfig): Configuration for the regression head.
            input_dim (int): Input dimension for the regression head.

        Returns:
            nn.Module: The constructed regression head.
        """
        head = nn.Sequential()

        head.add_module("initial_norm", nn.LayerNorm(head_config.input_dimensions))

        dimensions = [head_config.input_dimensions, *head_config.hidden_dimensions]

        for idx, (in_dim, out_dim) in enumerate(pairwise(dimensions)):
            if head_config.head_type == HeadType.FULLY_CONNECTED:
                head.add_module(
                    f"fully_connected_{idx}",
                    FullyConnectedBlock(in_dim, out_dim, head_config.activation_fn),
                )
            elif head_config.head_type == HeadType.RESIDUAL:
                head.add_module(
                    f"residual_{idx}",
                    ResidualBlock(in_dim, out_dim, head_config.activation_fn),
                )
        head.add_module(
            f"linear_{idx + 1}",
            nn.Linear(dimensions[-1], 1),
        )

        return head

    def forward(self, descriptor, auxillary_data: dict | None = None):
        preds = []

        for name, head in self.task_heads.items():
            if auxillary_data is not None and name in auxillary_data:
                aux = auxillary_data[name].to(descriptor.device)
                input_data = torch.cat((descriptor, aux), dim=1)
                preds.append(head(input_data))
            else:
                preds.append(head(descriptor))
        # TODO: Make this return a dict of all tasks instead of a stacked tensor to ensure that the task predictions are returned in the correct order. This would require us to also change the way that the dataset yields the regression targets, would also be a dict then. Maybe it should be possible to just assert that the dataset task ordering and the model task ordering are identical.

        return preds


class MultiTaskRegressionModel(nn.Module):
    def __init__(
        self,
        regression_heads: MultitaskHeads,
        encoder: TransformerEncoder,
        preprocessor: AtomicDescriptorPreprocess,
        global_aggregator: GlobalAggregator,
    ):
        super().__init__()
        self.encoder = encoder
        self.multitask_heads = regression_heads
        self.preprocessor = preprocessor
        self.global_aggregator = global_aggregator

    def forward(self, x, padding_mask=None, auxillary_data: dict | None = None):
        descriptor = self.get_molecular_descriptor(x, padding_mask)

        preds = self.multitask_heads(descriptor, auxillary_data)
        # Compute outputs from each head.
        # Each head's output is assumed to be of shape (batch_size, output_dim)

        # Concatenate outputs along the feature dimension.
        # Final shape: (batch_size, N_tasks * output_dim)
        pred = torch.cat(preds, dim=-1)
        return pred

    def get_molecular_descriptor(self, x, padding_mask=None) -> torch.Tensor:
        x = self.preprocessor(x)
        x = self.encoder(x, padding_mask)
        descriptor = self.global_aggregator(x)

        return descriptor


class FusedMultiHeadRegression(nn.Module):
    # A class that implements the fused calulation of regression heads by using the torch.bmm (batched matrix multiply) instead of sequentially calculating each head.
    pass
