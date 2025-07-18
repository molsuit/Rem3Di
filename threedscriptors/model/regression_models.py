from collections import OrderedDict
from itertools import pairwise

import torch
import torch.nn as nn

from threedscriptors.configuration.architecture_config import (
    HeadType,
    RegressionHeadConfig,
)
from threedscriptors.data_handling.sample import Sample


from threedscriptors.model.model_output import ModelOutput
from threedscriptors.model.preprocessing.preprocessing import Preprocessor
from threedscriptors.model.encoder import TransformerEncoder
from threedscriptors.model.pair_encoder import TransformerPairEncoder

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
                    ("activation", nn.SiLU()),
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        out = self.block(x)
        return out


class RegressionHead(nn.Module):


    def __init__(self, head_config : RegressionHeadConfig):

        """
        Build a regression head based on the provided configuration.

        Args:
            head_config (RegressionHeadConfig): Configuration for the regression head.
            input_dim (int): Input dimension for the regression head.

        Returns:
            nn.Module: The constructed regression head.
        """

        super().__init__()

        self.head_config = head_config
        self.task_config = head_config.task_config

        head = nn.Sequential()

        head.add_module("initial_norm", nn.LayerNorm(head_config.input_dimensions))

        dimensions = [head_config.input_dimensions, *head_config.hidden_dimensions]
        idx = 0
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

        self.head = head
        
        

        self.register_buffer("task_mean", self.task_config.mean)
        self.register_buffer("task_std", self.task_config.std)


    def forward(self, molecular_descriptor):
        return self.head(molecular_descriptor)
        

    def inference(self, molecular_descriptor):
        standardized_prediction = self.head(molecular_descriptor)
            # undo the standardization:
        return (standardized_prediction * self.task_std) + self.task_mean




class MultitaskHeads(nn.Module):
    def __init__(self, regression_head_configs: list[RegressionHeadConfig]):
        super().__init__()
        self.regression_head_configs = regression_head_configs

        self.task_list = [conf.task_name for conf in regression_head_configs]
        self.N_tasks = len(self.task_list)

        self.task_heads = nn.ModuleDict({
            conf.task_name: RegressionHead(conf)
            for conf in regression_head_configs
        })
   

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

        preds = torch.cat(preds, dim=-1)

        return preds


    def inference(self, descriptor, auxillary_data: dict | None = None):
        preds = []

        for name, head in self.task_heads.items():
            if auxillary_data is not None and name in auxillary_data:
                aux = auxillary_data[name].to(descriptor.device)
                input_data = torch.cat((descriptor, aux), dim=1)
                preds.append(head.inference(input_data))
            else:
                preds.append(head.inference(descriptor))

        preds = torch.cat(preds, dim=-1)

        return preds


class FusedMultiHeadRegression(nn.Module):
    # A class that implements the fused calulation of regression heads by using the torch.bmm (batched matrix multiply) instead of sequentially calculating each head.
    pass


class MultiTaskRegressionModel(nn.Module):
    def __init__(
        self,
        preprocessor: Preprocessor,
        encoder: TransformerEncoder | TransformerPairEncoder,
        regression_heads: MultitaskHeads,
    ):
        super().__init__()
        self.encoder = encoder
        self.multitask_heads = regression_heads
        self.preprocessor = preprocessor

    def forward(self, sample: Sample):
        molecular_descriptor = self.get_molecular_descriptor(sample)

        preds = self.multitask_heads(molecular_descriptor, sample.auxillary_data)

        return ModelOutput(
            molecular_descriptor=molecular_descriptor, regression_predictions=preds
        )

    def get_molecular_descriptor(self, sample: Sample) -> torch.Tensor:
        preprocessed_sample = self.preprocessor(sample)
        molecular_descriptor = self.encoder(preprocessed_sample)
        return molecular_descriptor

